# DocChat — PDF Question Answering with RAG

A locally-running RAG (Retrieval-Augmented Generation) system that lets you upload any PDF and ask questions about it in natural language. No cloud APIs, no data leaving your machine — everything runs locally via Ollama.

---

## How It Works

```
PDF Upload
    ↓
PyPDFLoader → extract text page by page
    ↓
RecursiveCharacterTextSplitter → chunks (1000 tokens, 200 overlap)
    ↓
HuggingFace Embeddings (all-MiniLM-L6-v2) → vector representations
    ↓
ChromaDB → persisted vector store
    ↓
User Question → Max Marginal Relevance retrieval (k=5, fetch_k=12)
    ↓
RAG prompt (context + question) → Ollama local LLM
    ↓
Answer + source page citations
```

**Why MMR over plain similarity search?** Max Marginal Relevance retrieves chunks that are both relevant *and* diverse — avoiding redundant context when the same fact appears multiple times across a document.

---

## Features

- Upload any PDF and chat with it instantly
- Fully local — no OpenAI key, no internet required for inference
- Switch between models at runtime (sidebar dropdown)
- Shows page-level source citations with every answer
- Text preview of extracted PDF content
- Persistent vector store (ChromaDB) — reloads without re-embedding
- Clean dark UI built with Streamlit

---

## Project Structure

```
├── app.py                  # Streamlit UI + session state
├── src/
│   ├── pdf_loader.py       # PDF loading via PyPDFLoader
│   ├── chunker.py          # Text splitting (RecursiveCharacterTextSplitter)
│   ├── embeddings.py       # HuggingFace embeddings + ChromaDB vectorstore
│   ├── retriever.py        # MMR retrieval
│   └── rag_chain.py        # Prompt building + Ollama inference
├── data/                   # Uploaded PDFs (gitignored)
├── chroma_db/              # Persisted vector store (gitignored)
└── requirements.txt
```

---

## Setup

### Prerequisites
- Python 3.9+
- [Ollama](https://ollama.com/) installed and running locally

### Install dependencies
```bash
pip install -r requirements.txt
```

### Pull a model via Ollama
```bash
ollama pull qwen2.5:7b
# or any other supported model
```

### Run
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Supported Models (via Ollama)

Selectable from the sidebar at runtime:

| Model | Size |
|-------|------|
| qwen3.5:9b | ~6GB |
| llama3.1:8b | ~5GB |
| qwen3:8b | ~5GB |
| qwen2.5:7b | ~5GB |
| phi3:mini | ~2GB |

Any model available in `ollama list` can be added to the dropdown in `app.py`.

---

## Tech Stack

- **UI:** Streamlit
- **PDF parsing:** LangChain + PyPDFLoader
- **Chunking:** RecursiveCharacterTextSplitter
- **Embeddings:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
- **Vector store:** ChromaDB (persisted)
- **Retrieval:** Max Marginal Relevance search
- **LLM inference:** Ollama (local, no API key needed)
- **LLM communication:** Direct HTTP to Ollama API (`urllib`, no LangChain dependency for inference)
