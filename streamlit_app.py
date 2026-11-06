import os
import hashlib

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from tavily import TavilyClient

from rag_engine import (
    load_embedding_model,
    process_pdf,
    search_document,
    generate_answer,
    generate_followup_questions,
    search_web,
    generate_web_answer,
    generate_combined_answer
)


# ==========================================
# 1. CONFIGURATION
# ==========================================

load_dotenv()

st.set_page_config(
    page_title="KNOWVA | Document Intelligence",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

GROQ_MODEL = "openai/gpt-oss-20b"


# ==========================================
# 2. SESSION STATE
# ==========================================

session_defaults = {
    "chunks": None,
    "faiss_index": None,
    "document_name": None,
    "document_hash": None,
    "chat_history": [],
    "question_input": "",
    "submit_question": False
}

for key, value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ==========================================
# 3. LOAD EMBEDDING MODEL
# ==========================================

@st.cache_resource
def get_embedding_model():
    return load_embedding_model()


# ==========================================
# 4. FOLLOW-UP QUESTION CALLBACK
# ==========================================

def select_followup_question(selected_question):
    """Automatically submit a selected follow-up question."""

    st.session_state.question_input = selected_question
    st.session_state.submit_question = True


# ==========================================
# 5. CUSTOM DESIGN
# ==========================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

.stApp {
    background-color: #F7F5F0;
    color: #202D35;
    font-family: 'DM Sans', sans-serif;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}

[data-testid="stSidebar"] {
    background-color: #202D35;
}

[data-testid="stSidebar"] * {
    color: #F7F5F0;
}

.brand {
    font-family: 'Playfair Display', serif;
    font-size: 34px;
    font-weight: 700;
    color: #F7F5F0;
    letter-spacing: 1px;
}

.brand-subtitle {
    color: #B9C8C2;
    font-size: 12px;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.sidebar-label {
    color: #AABBB4;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 2px;
    margin-top: 35px;
}

.hero {
    background: #E9EDE6;
    padding: 36px;
    border-radius: 18px;
    border: 1px solid #DCE3D9;
    margin-bottom: 28px;
}

.hero-tag {
    color: #648477;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 39px;
    line-height: 1.2;
    color: #202D35;
    margin-top: 12px;
}

.hero-description {
    color: #61716D;
    font-size: 15px;
    line-height: 1.8;
    max-width: 650px;
}

.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 25px;
    color: #202D35;
    font-weight: 600;
}

.info-card {
    background: #FFFFFF;
    padding: 22px;
    border: 1px solid #E8E5DC;
    border-radius: 14px;
    min-height: 145px;
}

.info-card h3 {
    color: #30443D;
    font-size: 16px;
}

.info-card p {
    color: #77827D;
    font-size: 13px;
    line-height: 1.6;
}

.upload-note {
    background: #EEF1EB;
    border-left: 4px solid #789789;
    padding: 15px;
    border-radius: 5px;
    color: #52665D;
    font-size: 13px;
}

div[data-testid="stFileUploader"] {
    background: #FFFFFF;
    border: 1px dashed #A6B9AA;
    border-radius: 12px;
    padding: 18px;
}

.stButton button {
    background: #526F62;
    color: white;
    border-radius: 9px;
    border: none;
    padding: 10px 22px;
    font-weight: 600;
}

.stButton button:hover {
    background: #3E594D;
    color: white;
}

.answer-card {
    background: #FFFFFF;
    padding: 24px;
    border: 1px solid #E8E5DC;
    border-radius: 14px;
}

hr {
    border-color: #E4E2DA;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 6. SIDEBAR
# ==========================================

# Keep the sidebar background, but do not display branding text.
with st.sidebar:
    st.empty()


# IMPORTANT: Keep this outside the sidebar.
# This prevents the NameError and keeps Overview active.
page = "⌂  Overview"


# ==========================================
# 7. MAIN DASHBOARD
# ==========================================

st.markdown("""
<div class="hero">
    <div class="hero-tag">YOUR PERSONAL RESEARCH SPACE</div>
    <div class="hero-title">Knowledge begins with a question.</div>
    <div class="hero-description">
        Upload a document, explore its information, and discover
        connections beyond its pages. Your AI-powered research
        workspace starts here.
    </div>
</div>
""", unsafe_allow_html=True)


if page != "⌂  Overview":

    st.markdown(
        '<div class="section-title">Coming soon</div>',
        unsafe_allow_html=True
    )

    st.info(
        "This workspace feature will be developed in a later stage. "
        "Your document and external research workspace is available in Overview."
    )

    st.stop()


# ==========================================
# 8. WORKSPACE OVERVIEW
# ==========================================

st.markdown(
    '<div class="section-title">Explore your workspace</div>',
    unsafe_allow_html=True
)

st.write("Everything you need to turn documents into knowledge.")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-card">
        <h3>▤ Document Intelligence</h3>
        <p>Upload PDFs and ask questions about their content.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <h3>✦ Deep Research</h3>
        <p>Explore additional information from external web sources.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-card">
        <h3>◈ Knowledge Discovery</h3>
        <p>Discover insights by combining document and web information.</p>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.markdown("---")


# ==========================================
# 9. RESEARCH MODE
# ==========================================

st.markdown(
    '<div class="section-title">Choose your research mode</div>',
    unsafe_allow_html=True
)

research_mode = st.radio(
    "Research mode",
    [
        "Document only",
        "External research",
        "Combined research"
    ],
    horizontal=True,
    help=(
        "Document only uses your PDF. External research searches the web. "
        "Combined research uses both sources."
    )
)

if research_mode == "Document only":
    st.caption(
        "Answers are generated using information from your uploaded PDF."
    )

elif research_mode == "External research":
    st.caption(
        "Searches external websites. A PDF upload is not required."
    )

else:
    st.caption(
        "Uses your uploaded PDF together with external web research."
    )


# ==========================================
# 10. DOCUMENT UPLOAD AND PROCESSING
# ==========================================

st.write("")
st.markdown("---")

st.markdown(
    '<div class="section-title">Start with a document</div>',
    unsafe_allow_html=True
)

if research_mode == "External research":
    st.write("PDF upload is optional for external research.")

else:
    st.write("Upload a PDF to begin exploring its information.")

uploaded_file = st.file_uploader(
    "Choose your PDF document",
    type=["pdf"],
    help="Upload a PDF document for document or combined research."
)

if uploaded_file is not None:

    pdf_bytes = uploaded_file.getvalue()

    current_hash = hashlib.sha256(pdf_bytes).hexdigest()

    st.success(f"Document selected: {uploaded_file.name}")

    st.caption(
        f"File size: {uploaded_file.size / 1024:.2f} KB"
    )

    if current_hash != st.session_state.document_hash:

        with st.spinner("Processing your document..."):

            try:

                embedding_model = get_embedding_model()

                chunks, index = process_pdf(
                    pdf_bytes,
                    embedding_model
                )

                st.session_state.chunks = chunks
                st.session_state.faiss_index = index
                st.session_state.document_name = uploaded_file.name
                st.session_state.document_hash = current_hash
                st.session_state.chat_history = []
                st.session_state.question_input = ""

                st.success(
                    f"Document processed successfully! "
                    f"{len(chunks)} searchable chunks created."
                )

            except Exception as error:

                st.error(f"Unable to process the PDF: {error}")

                st.stop()

    else:

        st.success("Your document is ready for questions.")

    st.markdown("""
    <div class="upload-note">
        Your document has been converted into searchable information.
        Ask questions to discover answers from its content.
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# 11. QUESTION WORKSPACE
# ==========================================

st.write("")
st.markdown("---")

st.markdown(
    '<div class="section-title">What would you like to discover?</div>',
    unsafe_allow_html=True
)

question = st.text_input(
    "Your question",
    placeholder="Ask a question, explore an idea, or discover something new...",
    key="question_input"
)

start_clicked = st.button(
    "Start exploring",
    use_container_width=True
)

# Follow-up question buttons use this flag to submit automatically.
auto_submit = st.session_state.submit_question
st.session_state.submit_question = False

should_submit = start_clicked or auto_submit


# ==========================================
# 12. QUESTION PROCESSING
# ==========================================

if should_submit:

    question = st.session_state.question_input.strip()

    if not question:

        st.warning("Please enter your question.")

    elif research_mode in ["Document only", "Combined research"] and (
        st.session_state.chunks is None
    ):

        st.warning("Please upload and process a PDF document first.")

    elif not GROQ_API_KEY:

        st.error("Groq API key not found. Please check your .env file.")

    elif research_mode in ["External research", "Combined research"] and (
        not TAVILY_API_KEY
    ):

        st.error("Tavily API key not found. Please check your .env file.")

    else:

        try:

            groq_client = Groq(api_key=GROQ_API_KEY)

            answer = ""
            source_pages = []
            web_sources = []
            followup_questions = []

            # ----------------------------------
            # DOCUMENT ONLY
            # ----------------------------------

            if research_mode == "Document only":

                embedding_model = get_embedding_model()

                with st.spinner("Searching your document..."):

                    retrieved_chunks = search_document(
                        question,
                        st.session_state.chunks,
                        st.session_state.faiss_index,
                        embedding_model,
                        top_k=3
                    )

                if not retrieved_chunks:

                    st.warning(
                        "No relevant information was found in the document."
                    )

                else:

                    with st.spinner("Generating your answer with AI..."):

                        answer = generate_answer(
                            question,
                            retrieved_chunks,
                            groq_client,
                            model_name=GROQ_MODEL
                        )

                    source_pages = sorted({
                        chunk["page"] for chunk in retrieved_chunks
                    })

                    try:

                        with st.spinner("Generating follow-up questions..."):

                            followup_questions = generate_followup_questions(
                                question,
                                retrieved_chunks,
                                groq_client,
                                model_name=GROQ_MODEL
                            )

                    except Exception as followup_error:

                        st.warning(
                            "The answer is ready, but follow-up questions "
                            f"could not be generated: {followup_error}"
                        )

            # ----------------------------------
            # EXTERNAL RESEARCH
            # ----------------------------------

            elif research_mode == "External research":

                tavily_client = TavilyClient(
                    api_key=TAVILY_API_KEY
                )

                with st.spinner(
                    "Searching external websites with Tavily..."
                ):

                    web_sources = search_web(
                        question,
                        tavily_client,
                        max_results=5
                    )

                if not web_sources:

                    st.warning(
                        "No useful external web sources were found."
                    )

                else:

                    with st.spinner("Generating your research answer..."):

                        answer = generate_web_answer(
                            question,
                            web_sources,
                            groq_client,
                            model_name=GROQ_MODEL
                        )

            # ----------------------------------
            # COMBINED RESEARCH
            # ----------------------------------

            elif research_mode == "Combined research":

                embedding_model = get_embedding_model()

                with st.spinner("Searching your uploaded document..."):

                    retrieved_chunks = search_document(
                        question,
                        st.session_state.chunks,
                        st.session_state.faiss_index,
                        embedding_model,
                        top_k=3
                    )

                source_pages = sorted({
                    chunk["page"] for chunk in retrieved_chunks
                })

                tavily_client = TavilyClient(
                    api_key=TAVILY_API_KEY
                )

                with st.spinner(
                    "Searching external websites with Tavily..."
                ):

                    web_sources = search_web(
                        question,
                        tavily_client,
                        max_results=5
                    )

                if not retrieved_chunks and not web_sources:

                    st.warning(
                        "No useful information was found in either source."
                    )

                else:

                    with st.spinner(
                        "Combining document and web research..."
                    ):

                        answer = generate_combined_answer(
                            question,
                            retrieved_chunks,
                            web_sources,
                            groq_client,
                            model_name=GROQ_MODEL
                        )

                    if retrieved_chunks:

                        try:

                            followup_questions = generate_followup_questions(
                                question,
                                retrieved_chunks,
                                groq_client,
                                model_name=GROQ_MODEL
                            )

                        except Exception as followup_error:

                            st.warning(
                                "Follow-up questions could not be generated: "
                                f"{followup_error}"
                            )

            # ----------------------------------
            # SAVE RESEARCH RESULT
            # ----------------------------------

            if answer:

                st.session_state.chat_history.append({
                    "question": question,
                    "mode": research_mode,
                    "answer": answer,
                    "pages": source_pages,
                    "web_sources": web_sources,
                    "followups": followup_questions
                })

        except Exception as error:

            st.error(
                "Unable to complete your research. "
                f"Error details: {error}"
            )


# ==========================================
# 13. DISPLAY ANSWERS AND SOURCES
# ==========================================

if st.session_state.chat_history:

    st.write("")
    st.markdown("---")

    st.markdown(
        '<div class="section-title">Your research results</div>',
        unsafe_allow_html=True
    )

    for history_number, item in enumerate(
        reversed(st.session_state.chat_history)
    ):

        with st.container():

            st.markdown(f"**Research mode:** {item['mode']}")

            st.markdown("**Your question**")
            st.write(item["question"])

            st.markdown("**KNOWVA's answer**")

            with st.container(border=True):
                st.markdown(item["answer"])

            # ----------------------------------
            # PDF SOURCE REFERENCES
            # ----------------------------------

            if item.get("pages"):

                st.markdown("**PDF source pages**")

                page_labels = ", ".join(
                    [f"Page {page}" for page in item["pages"]]
                )

                st.caption(page_labels)

            # ----------------------------------
            # WEB SOURCE REFERENCES
            # ----------------------------------

            if item.get("web_sources"):

                st.markdown("**External web sources**")

                for source_number, source in enumerate(
                    item["web_sources"],
                    start=1
                ):

                    st.markdown(
                        f"**Web Source {source_number}: "
                        f"{source['title']}**"
                    )

                    st.markdown(
                        f"[Open source website]({source['url']})"
                    )

                    st.caption(source["content"][:350] + "...")

            # ----------------------------------
            # FOLLOW-UP QUESTIONS
            # ----------------------------------

            followups = item.get("followups", [])

            if followups:

                st.markdown("**Explore these next questions**")

                st.caption(
                    "Choose a question to continue exploring your document."
                )

                for question_number, followup in enumerate(followups):

                    st.button(
                        followup,
                        key=f"followup_{history_number}_{question_number}",
                        use_container_width=True,
                        on_click=select_followup_question,
                        args=(followup,)
                    )

            st.divider()


# ==========================================
# 14. FOOTER
# ==========================================

st.write("")
st.markdown("---")

st.markdown("""
<div style="text-align:center; color:#89948D;
            font-size:12px; padding:15px;">
    KNOWVA · AI Document & Research Intelligence
    <br>
    Turn information into understanding.
</div>
""", unsafe_allow_html=True)