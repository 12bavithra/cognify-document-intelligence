
import numpy as np
import pymupdf
import faiss

from sentence_transformers import SentenceTransformer


# ==========================================
# 1. LOAD EMBEDDING MODEL
# ==========================================

def load_embedding_model():
    """Load the local Sentence Transformer model."""

    return SentenceTransformer("all-MiniLM-L6-v2")


# ==========================================
# 2. PROCESS UPLOADED PDF
# ==========================================

def process_pdf(pdf_bytes, model):
    """
    Extract PDF text, create chunks,
    generate embeddings and build FAISS index.
    """

    document = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    chunks = []
    chunk_size = 500
    overlap = 100

    try:
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

    finally:
        document.close()

    if not chunks:
        raise ValueError("No readable text found in this PDF.")

    chunk_texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        chunk_texts,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    embeddings = embeddings.astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return chunks, index


# ==========================================
# 3. RETRIEVE RELEVANT DOCUMENT INFORMATION
# ==========================================

def search_document(query, chunks, index, model, top_k=3):
    """Find the most relevant document chunks."""

    if not query.strip() or index is None or index.ntotal == 0:
        return []

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
# 4. GENERATE ANSWER USING GROQ
# ==========================================

def generate_answer(
    query,
    retrieved_chunks,
    client,
    model_name="openai/gpt-oss-20b"
):
    """Generate an answer using retrieved PDF content."""

    if not retrieved_chunks:
        return "No relevant information was found."

    context = ""

    for i, chunk in enumerate(retrieved_chunks, start=1):
        context += (
            f"\nSource {i} - Page {chunk['page']}:\n"
            f"{chunk['text']}\n"
        )

    prompt = f"""
You are KNOWVA, an AI Document Research Assistant.

Answer the user's question using only the
document context provided below.

Instructions:
1. Give a clear and simple answer.
2. Do not invent information.
3. If the document does not contain enough
   information, clearly say so.
4. Do not use outside knowledge.
5. Keep the answer relevant to the question.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{query}
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_completion_tokens=1000
    )

    return response.choices[0].message.content


# ==========================================
# 5. GENERATE FOLLOW-UP QUESTIONS
# ==========================================

def generate_followup_questions(
    query,
    retrieved_chunks,
    client,
    model_name="openai/gpt-oss-20b"
):
    """Generate three follow-up questions from PDF information."""

    if not retrieved_chunks:
        return []

    context = ""

    for chunk in retrieved_chunks:
        context += (
            f"\nPage {chunk['page']}:\n"
            f"{chunk['text']}\n"
        )

    prompt = f"""
You are KNOWVA, an AI Document Research Assistant.

Generate exactly THREE useful follow-up questions
based on the document context.

Rules:
1. Questions must be relevant to the document.
2. Explore different aspects of the information.
3. Do not repeat the original question.
4. Do not invent unsupported topics.
5. Keep questions short and easy to understand.
6. Return only three numbered questions.
7. Do not include answers or explanations.

DOCUMENT CONTEXT:
{context}

USER'S ORIGINAL QUESTION:
{query}
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_completion_tokens=300
    )

    result = response.choices[0].message.content

    questions = []

    for line in result.splitlines():
        line = line.strip()

        if not line:
            continue

        cleaned = line.lstrip("0123456789").lstrip(".)- ").strip()

        if cleaned.endswith("?"):
            questions.append(cleaned)

    return questions[:3]


# ==========================================
# 6. EXTERNAL WEB RESEARCH WITH TAVILY
# ==========================================

def search_web(query, tavily_client, max_results=5):
    """Search the web and return source details."""

    if not query or not query.strip():
        return []

    response = tavily_client.search(
        query=query.strip(),
        search_depth="basic",
        max_results=max_results,
        include_answer=False
    )

    results = []

    for item in response.get("results", []):
        title = item.get("title") or "Untitled source"
        url = item.get("url") or ""
        content = item.get("content") or ""

        if url and content:
            results.append({
                "title": title,
                "url": url,
                "content": content
            })

    return results


# ==========================================
# 7. GENERATE ANSWER FROM WEB SOURCES
# ==========================================

def generate_web_answer(
    query,
    web_results,
    client,
    model_name="openai/gpt-oss-20b"
):
    """Generate an answer using retrieved web search results."""

    if not web_results:
        return (
            "I couldn't find useful external sources for this question. "
            "Try rephrasing it or using a more specific search query."
        )

    context_parts = []

    for number, source in enumerate(web_results, start=1):
        context_parts.append(
            f"Web Source {number}\n"
            f"Title: {source['title']}\n"
            f"URL: {source['url']}\n"
            f"Content: {source['content']}\n"
        )

    context = "\n".join(context_parts)

    prompt = f"""
You are KNOWVA, an AI Document and Research Assistant.

Answer the user's question using ONLY the web search source content.

Rules:
1. Give a clear and useful answer in simple language.
2. Do not add unsupported facts.
3. If sources do not provide enough information, say so.
4. Add references such as [Web Source 1].
5. Do not invent URLs, titles, dates or statistics.
6. If sources disagree, explain the disagreement.

WEB SEARCH RESULTS:
{context}

USER QUESTION:
{query}
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_completion_tokens=1200
    )

    return response.choices[0].message.content


# ==========================================
# 8. COMBINED DOCUMENT + WEB RESEARCH
# ==========================================

def generate_combined_answer(
    query,
    retrieved_chunks,
    web_results,
    client,
    model_name="openai/gpt-oss-20b"
):
    """Generate an answer using both PDF and web information."""

    document_context = ""

    for number, chunk in enumerate(retrieved_chunks, start=1):
        document_context += (
            f"Document Source {number} - PDF Page {chunk['page']}:\n"
            f"{chunk['text']}\n\n"
        )

    web_context_parts = []

    for number, source in enumerate(web_results, start=1):
        web_context_parts.append(
            f"Web Source {number}\n"
            f"Title: {source['title']}\n"
            f"URL: {source['url']}\n"
            f"Content: {source['content']}\n"
        )

    web_context = "\n".join(web_context_parts)

    prompt = f"""
You are KNOWVA, an AI Document and Research Assistant.

Answer the question using both the uploaded document
and the external web search results.

Important rules:
1. Clearly separate document information and web information.
2. Use labels such as "From the uploaded document"
   and "From external research".
3. Cite document information using PDF page numbers.
4. Cite web information using [Web Source 1] references.
5. Do not invent facts or citations.
6. If sources disagree, explain the difference.
7. If either source lacks relevant information, say so.
8. Keep the answer clear and easy to understand.

UPLOADED DOCUMENT CONTEXT:
{document_context if document_context else "No relevant PDF passages were retrieved."}

EXTERNAL WEB SOURCES:
{web_context if web_context else "No useful web results were retrieved."}

USER QUESTION:
{query}
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_completion_tokens=1500
    )

    return response.choices[0].message.content
