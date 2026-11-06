
import os
import numpy as np
import pymupdf
import faiss

from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError(
        "GROQ_API_KEY not found. Please check your .env file."
    )

client = Groq(api_key=groq_api_key)

MODEL_NAME = "openai/gpt-oss-20b"


# ==========================================
# 2. PDF DOCUMENT READING
# ==========================================

pdf_path = "documents/data_analysis.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

document = pymupdf.open(pdf_path)

print("Total pages:", len(document))

chunks = []

chunk_size = 500
overlap = 100

for page_number, page in enumerate(document, start=1):

    text = page.get_text()

    start = 0

    while start < len(text):

        end = start + chunk_size
        chunk_text = text[start:end]

        if chunk_text.strip():
            chunks.append({
                "page": page_number,
                "text": chunk_text
            })

        start += chunk_size - overlap

document.close()

if not chunks:
    raise ValueError("No readable text found in the PDF.")

print("Total chunks created:", len(chunks))


# ==========================================
# 3. LOAD THE EMBEDDING MODEL
# ==========================================

print("\nLoading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded successfully!")


# ==========================================
# 4. GENERATE EMBEDDINGS
# ==========================================

chunk_texts = [chunk["text"] for chunk in chunks]

embeddings = model.encode(
    chunk_texts,
    convert_to_numpy=True,
    show_progress_bar=True
)

embeddings = embeddings.astype("float32")

print("\nEmbeddings created successfully!")
print("Number of embeddings:", len(embeddings))
print("Embedding dimensions:", embeddings.shape)


# ==========================================
# 5. CREATE FAISS INDEX
# ==========================================

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("\nFAISS index created successfully!")
print("Total vectors stored:", index.ntotal)


# ==========================================
# 6. RETRIEVE RELEVANT INFORMATION
# ==========================================

def search_document(query, top_k=3):

    if not query.strip():
        print("Please enter a valid question.")
        return

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        min(top_k, index.ntotal)
    )

    retrieved_chunks = []

    for idx in indices[0]:

        if idx == -1:
            continue

        retrieved_chunks.append(chunks[idx])

    return retrieved_chunks


# ==========================================
# 7. GENERATE ANSWER USING GROQ
# ==========================================

def generate_answer(query, retrieved_chunks):

    context = ""

    for i, chunk in enumerate(retrieved_chunks, start=1):

        context += (
            f"\nSource {i} - Page {chunk['page']}:\n"
            f"{chunk['text']}\n"
        )

    prompt = f"""
You are an AI Document Assistant.

Answer the user's question using only the information
provided in the document context below.

Instructions:
1. Give a clear and simple answer.
2. Do not invent information.
3. If the answer is not available in the context,
   clearly say that the document does not provide
   enough information to answer.
4. Do not use outside knowledge.
5. Keep the answer relevant to the user's question.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{query}
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_completion_tokens=1000
    )

    return response.choices[0].message.content


# ==========================================
# 8. ASK QUESTIONS
# ==========================================

print("\nAI Document Assistant is ready!")

while True:

    user_query = input(
        "\nAsk a question about the PDF "
        "(or type 'exit' to stop): "
    )

    if user_query.lower().strip() == "exit":

        print("\nThank you for using AI Document Assistant!")
        break

    retrieved_chunks = search_document(user_query, top_k=3)

    if not retrieved_chunks:
        print("No relevant information found.")
        continue

    print("\nGenerating answer using Groq...")

    try:

        answer = generate_answer(
            user_query,
            retrieved_chunks
        )

        print("\n========== ANSWER ==========")
        print(answer)

        print("\n========== SOURCE PAGES ==========")

        source_pages = sorted(
            set(chunk["page"] for chunk in retrieved_chunks)
        )

        for page in source_pages:
            print(f"Page {page}")

        print("==============================")

    except Exception as error:

        print("\nUnable to generate an answer.")
        print("Error:", error)
