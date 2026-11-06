# AI Document Intelligence Platform

An AI-powered document assistant that helps users understand PDF documents, retrieve relevant information, and explore topics through external web research.

## 🚀 Live Demo

[Try AI Document Intelligence Platform](https://cognify-document-intelligence.streamlit.app/)

## 📌 Project Overview

The AI Document Intelligence Platform uses Retrieval-Augmented Generation (RAG) to answer questions based on uploaded PDF documents.

Users can upload a PDF, ask questions about its content, and receive context-aware answers. The application also supports external web research and combined research using both document content and web sources.

## ✨ Key Features

- **PDF Upload:** Upload and process PDF documents.
- **Document-Based Question Answering:** Ask questions and get answers from uploaded documents.
- **External Web Research:** Retrieve additional information from web sources.
- **Combined Research:** Generate answers using both document content and external research.
- **Source References:** View relevant document page references.
- **Follow-up Questions:** Explore related questions based on previous queries.
- **Interactive Interface:** User-friendly interface built with Streamlit.

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| PDF Processing | PyMuPDF |
| Embedding Model | Sentence Transformers (all-MiniLM-L6-v2) |
| Vector Database | FAISS |
| LLM | Groq API |
| Web Search | Tavily API |
| Environment Management | Python dotenv |

## ⚙️ How It Works

1. The user uploads a PDF document.
2. The application extracts text from the PDF.
3. The extracted content is divided into smaller chunks.
4. Sentence Transformers converts the chunks into vector embeddings.
5. FAISS stores and retrieves relevant information.
6. The retrieved context is passed to the language model.
7. Groq generates a context-aware response.
8. Tavily provides external web research when required.

## 🔍 Research Modes

**1. Document Only**

Answers questions using information from the uploaded PDF.

**2. External Research**

Uses external web sources to gather information.

**3. Combined Research**

Combines relevant PDF content with external web information to generate answers.

## 💻 Run Locally

**1. Clone the repository**

```bash
git clone https://github.com/12bavithra/cognify-document-intelligence.git
```

**2. Navigate to the project folder**

```bash
cd cognify-document-intelligence
```

**3. Create and activate a Python 3.11 virtual environment**

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

**4. Install dependencies**

```bash
pip install -r requirements.txt
```

**5. Configure API keys**

Create a `.env` file in the project root directory:

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

**6. Run the application**

```bash
streamlit run streamlit_app.py
```

## 🔐 Environment Variables

The application requires:

- `GROQ_API_KEY` – For AI-generated responses.
- `TAVILY_API_KEY` – For external web research.

Keep API keys private. Do not upload your `.env` file to GitHub.

## 🌐 Deployment

Deployed using Streamlit Community Cloud.

**Live Application:** https://cognify-document-intelligence.streamlit.app/



*An AI-powered approach to making document research simpler and more accessible.*
