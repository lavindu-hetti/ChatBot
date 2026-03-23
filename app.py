import os
import streamlit as st
from src.chunker import build_text_preview, chunk_documents
from src.embeddings import build_vectorstore
from src.pdf_loader import load_pdf, save_uploaded_file
from src.rag_chain import answer_with_retrieval

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="DocChat", page_icon="📄", layout="wide")

# ----------------------------
# Custom CSS
# ----------------------------
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

  /* ── Global reset ── */
  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
  }

  /* ── Hide default Streamlit chrome ── */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container {
    padding: 0 !important;
    max-width: 100% !important;
  }

  /* ── Root tokens ── */
  :root {
    --bg:        #0f1117;
    --surface:   #171923;
    --border:    #272d3d;
    --accent:    #4f8ef7;
    --accent-lo: rgba(79,142,247,.12);
    --text:      #e8ecf4;
    --muted:     #7a8299;
    --user-bg:   #1e2640;
    --ai-bg:     #171923;
    --danger:    #f76f6f;
    --success:   #4fcb8d;
    --radius:    12px;
  }

  /* ── Full-height shell ── */
  .stApp { background: var(--bg); color: var(--text); }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
    padding: 0 !important;
  }
  [data-testid="stSidebar"] > div { padding: 28px 20px 20px !important; }

  /* Sidebar brand */
  .sidebar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 28px;
  }
  .sidebar-brand .icon {
    width: 36px; height: 36px;
    background: var(--accent);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; flex-shrink: 0;
  }
  .sidebar-brand h1 {
    font-size: 18px !important;
    font-weight: 600 !important;
    color: var(--text) !important;
    margin: 0 !important; padding: 0 !important;
    line-height: 1 !important;
  }
  .sidebar-brand span {
    font-size: 11px;
    color: var(--muted);
    font-weight: 400;
  }

  /* Section labels */
  .sidebar-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: .08em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 8px;
  }

  /* Selectbox / uploader */
  [data-testid="stSelectbox"] label,
  [data-testid="stFileUploader"] label { display: none !important; }

  div[data-baseweb="select"] > div {
    background: var(--bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 13px !important;
  }
  div[data-baseweb="select"] svg { color: var(--muted) !important; }

  [data-testid="stFileUploadDropzone"] {
    background: var(--bg) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    transition: border-color .2s;
  }
  [data-testid="stFileUploadDropzone"]:hover {
    border-color: var(--accent) !important;
  }
  [data-testid="stFileUploadDropzone"] p,
  [data-testid="stFileUploadDropzone"] span {
    color: var(--muted) !important;
    font-size: 13px !important;
  }

  /* PDF status card */
  .pdf-card {
    background: var(--accent-lo);
    border: 1px solid rgba(79,142,247,.25);
    border-radius: var(--radius);
    padding: 14px 16px;
    margin-top: 18px;
  }
  .pdf-card .pdf-name {
    font-size: 13px;
    font-weight: 600;
    color: var(--accent);
    word-break: break-all;
    margin-bottom: 8px;
  }
  .pdf-card .pdf-meta {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
  }
  .pdf-card .meta-pill {
    font-size: 11px;
    background: rgba(255,255,255,.06);
    border-radius: 20px;
    padding: 3px 10px;
    color: var(--muted);
  }

  /* Clear button */
  [data-testid="stButton"] button {
    width: 100%;
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--muted) !important;
    border-radius: var(--radius) !important;
    font-size: 13px !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: 8px !important;
    transition: all .2s !important;
    margin-top: 8px;
  }
  [data-testid="stButton"] button:hover {
    border-color: var(--danger) !important;
    color: var(--danger) !important;
    background: rgba(247,111,111,.08) !important;
  }

  /* ── Main chat column ── */
  .chat-header {
    position: sticky;
    top: 0;
    z-index: 100;
    background: rgba(15,17,23,.92);
    backdrop-filter: blur(14px);
    border-bottom: 1px solid var(--border);
    padding: 16px 32px;
    display: flex;
    align-items: center;
    gap: 14px;
  }
  .chat-header .status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--muted);
    flex-shrink: 0;
  }
  .chat-header .status-dot.active {
    background: var(--success);
    box-shadow: 0 0 6px var(--success);
  }
  .chat-header .header-title { font-size: 15px; font-weight: 500; color: var(--text); }
  .chat-header .header-sub { font-size: 12px; color: var(--muted); margin-left: auto; }

  /* Chat area scroll wrapper */
  .chat-scroll { padding: 28px 32px 0; }

  /* Empty state */
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 52vh;
    text-align: center;
    color: var(--muted);
    gap: 12px;
  }
  .empty-state .empty-icon { font-size: 48px; opacity: .35; }
  .empty-state h3 { font-size: 17px; font-weight: 500; color: var(--text); opacity: .6; margin: 0; }
  .empty-state p { font-size: 13px; margin: 0; max-width: 280px; line-height: 1.6; }

  /* Message bubbles */
  [data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 6px 0 !important;
  }
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) .stMarkdown {
    background: var(--user-bg);
    border: 1px solid var(--border);
    border-radius: 14px 14px 4px 14px;
    padding: 12px 16px;
    font-size: 14px;
    line-height: 1.7;
    max-width: 75%;
    margin-left: auto;
  }
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) .stMarkdown {
    background: var(--ai-bg);
    border: 1px solid var(--border);
    border-radius: 14px 14px 14px 4px;
    padding: 12px 16px;
    font-size: 14px;
    line-height: 1.7;
    max-width: 80%;
  }

  [data-testid="stChatMessageAvatarUser"],
  [data-testid="stChatMessageAvatarAssistant"] {
    background: var(--border) !important;
    border-radius: 8px !important;
  }

  /* Chat input */
  [data-testid="stChatInput"] {
    background: var(--surface) !important;
    border-top: 1px solid var(--border) !important;
    padding: 16px 32px !important;
    position: sticky !important;
    bottom: 0 !important;
  }
  [data-testid="stChatInput"] textarea {
    background: var(--bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    resize: none !important;
    transition: border-color .2s !important;
  }
  [data-testid="stChatInput"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(79,142,247,.12) !important;
  }
  [data-testid="stChatInput"] button {
    background: var(--accent) !important;
    border-radius: 8px !important;
  }

  /* Alerts */
  [data-testid="stAlert"] {
    border-radius: var(--radius) !important;
    border: none !important;
    font-size: 13px !important;
  }

  /* Expander */
  [data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
  }
  [data-testid="stExpander"] summary {
    color: var(--muted) !important;
    font-size: 13px !important;
  }
  pre, code {
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
    background: var(--bg) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
  }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 5px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 99px; }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Session state
# ----------------------------
for key, default in [
    ("messages", []),
    ("uploaded_file_name", None),
    ("pdf_text_preview", ""),
    ("pdf_pages_count", 0),
    ("vectorstore", None),
    ("chunks_count", 0),
]:
    if key not in st.session_state:
        st.session_state[key] = default

os.makedirs("data", exist_ok=True)
os.makedirs("chroma_db", exist_ok=True)

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
      <div class="icon">📄</div>
      <div>
        <h1>DocChat</h1>
        <span>PDF Question Answering</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Model</div>', unsafe_allow_html=True)
    selected_model = st.selectbox(
        "model",
        ["qwen3.5:9b", "llama3.1:8b", "qwen3:8b", "qwen2.5:7b",
         "qwen2.5-coder:7b", "deepseek-coder:latest", "phi3:mini"],
        index=0,
    )

    st.markdown('<div class="sidebar-label" style="margin-top:20px">Document</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("pdf", type=["pdf"])

    if uploaded_file is not None:
        with st.spinner("Indexing document…"):
            file_path = save_uploaded_file(uploaded_file, data_dir="data")
            st.session_state.uploaded_file_name = uploaded_file.name
            docs = load_pdf(file_path)
            st.session_state.pdf_pages_count = len(docs)
            st.session_state.pdf_text_preview = build_text_preview(docs)
            chunks = chunk_documents(docs)
            st.session_state.chunks_count = len(chunks)
            vectorstore = build_vectorstore(chunks, persist_directory="chroma_db")
            st.session_state.vectorstore = vectorstore

    if st.session_state.uploaded_file_name:
        st.markdown(f"""
        <div class="pdf-card">
          <div class="pdf-name">📎 {st.session_state.uploaded_file_name}</div>
          <div class="pdf-meta">
            <span class="meta-pill">📄 {st.session_state.pdf_pages_count} pages</span>
            <span class="meta-pill">🧩 {st.session_state.chunks_count} chunks</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Preview extracted text"):
            if st.session_state.pdf_text_preview.strip():
                st.code(st.session_state.pdf_text_preview, language=None)
            else:
                st.warning("No text could be extracted from this PDF.")

    if st.button("🗑  Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ----------------------------
# Main: sticky header
# ----------------------------
is_ready = st.session_state.vectorstore is not None
dot_class = "status-dot active" if is_ready else "status-dot"
header_sub = f"Model: {selected_model}" if is_ready else "Upload a PDF to begin"

st.markdown(f"""
<div class="chat-header">
  <div class="{dot_class}"></div>
  <span class="header-title">{"Chat with your document" if is_ready else "No document loaded"}</span>
  <span class="header-sub">{header_sub}</span>
</div>
""", unsafe_allow_html=True)

# ----------------------------
# Chat messages
# ----------------------------
with st.container():
    st.markdown('<div class="chat-scroll">', unsafe_allow_html=True)

    if not st.session_state.messages:
        if is_ready:
            st.markdown("""
            <div class="empty-state">
              <div class="empty-icon">💬</div>
              <h3>Ask anything</h3>
              <p>Your document is indexed and ready. Type a question below.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="empty-state">
              <div class="empty-icon">📂</div>
              <h3>Upload a PDF</h3>
              <p>Use the sidebar to upload a PDF and start a conversation with your document.</p>
            </div>
            """, unsafe_allow_html=True)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# Chat input
# ----------------------------
prompt = st.chat_input(
    "Ask a question about your PDF…" if is_ready else "Upload a PDF first…",
    disabled=not is_ready,
)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(""):
            response = answer_with_retrieval(
                st.session_state.vectorstore,
                prompt,
                model_name=selected_model,
                k=3,
            )
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})