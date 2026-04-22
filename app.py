import streamlit as st
from rag_engine import RAGEngine
import os

st.set_page_config(
    page_title="DocMind — RAG PDF Chatbot",
    page_icon="🧠",
    layout="wide"
)

# ── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
    }
    .sub-header {
        color: #555;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .source-box {
        background: #f0f4ff;
        border-left: 3px solid #4f6ef7;
        padding: 0.6rem 1rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.82rem;
        color: #333;
        margin-top: 0.4rem;
    }
    .stat-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        text-align: center;
    }
    .stat-num { font-size: 1.5rem; font-weight: 700; color: #4f6ef7; }
    .stat-lbl { font-size: 0.75rem; color: #888; }
    .stChatMessage { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "rag" not in st.session_state:
    st.session_state.rag = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = {}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 DocMind")
    st.markdown("*RAG-powered PDF Intelligence*")
    st.divider()

    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="Paste your Gemini API key here",
        help="Get a free key at https://aistudio.google.com"
    )

    st.markdown("### Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files to query"
    )

    chunk_size = st.slider("Chunk size (tokens)", 300, 1000, 500, 50,
                           help="Larger chunks = more context per retrieval")
    top_k = st.slider("Top-K retrieval", 2, 8, 4,
                      help="Number of chunks retrieved per query")

    process_btn = st.button("⚡ Process Documents", use_container_width=True,
                            disabled=(not uploaded_files or not api_key))

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.markdown("**Built with**")
    st.markdown("LangChain · ChromaDB · Gemini API · Streamlit")
    st.markdown("**by [Aakash Bachhav](https://github.com/aakashbachhav)**")

# ── Document processing ───────────────────────────────────────────────────────
if process_btn and uploaded_files and api_key:
    with st.spinner("🔄 Processing documents — chunking, embedding, indexing..."):
        try:
            engine = RAGEngine(api_key=api_key, chunk_size=chunk_size, top_k=top_k)
            stats = engine.load_documents(uploaded_files)
            st.session_state.rag = engine
            st.session_state.doc_stats = stats
            st.session_state.chat_history = []
            st.success(f"✅ Ready! Indexed {stats['total_chunks']} chunks from {stats['num_docs']} document(s).")
        except Exception as e:
            st.error(f"Error processing documents: {e}")

# ── Main content ──────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🧠 DocMind — PDF Intelligence</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload PDFs, then ask anything. Answers are grounded in your documents with source citations.</p>', unsafe_allow_html=True)

# Stats row
if st.session_state.doc_stats:
    d = st.session_state.doc_stats
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="stat-card"><div class="stat-num">{d["num_docs"]}</div><div class="stat-lbl">Documents</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-card"><div class="stat-num">{d["total_pages"]}</div><div class="stat-lbl">Pages</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="stat-card"><div class="stat-num">{d["total_chunks"]}</div><div class="stat-lbl">Chunks indexed</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="stat-card"><div class="stat-num">{d["total_chunks"] * 500:,}</div><div class="stat-lbl">~Tokens stored</div></div>', unsafe_allow_html=True)
    st.markdown("")

# Chat display
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📄 Source chunks used", expanded=False):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(
                        f'<div class="source-box"><b>Source {i}</b> — {src["source"]}, page {src["page"]}<br>{src["snippet"]}</div>',
                        unsafe_allow_html=True
                    )

# Chat input
if prompt := st.chat_input("Ask anything about your documents...",
                            disabled=(st.session_state.rag is None)):
    # Show user message
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and reasoning..."):
            try:
                result = st.session_state.rag.query(prompt)
                st.markdown(result["answer"])
                if result["sources"]:
                    with st.expander("📄 Source chunks used", expanded=False):
                        for i, src in enumerate(result["sources"], 1):
                            st.markdown(
                                f'<div class="source-box"><b>Source {i}</b> — {src["source"]}, page {src["page"]}<br>{src["snippet"]}</div>',
                                unsafe_allow_html=True
                            )
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"]
                })
            except Exception as e:
                err = f"⚠️ Error: {e}"
                st.error(err)
                st.session_state.chat_history.append({"role": "assistant", "content": err})

# Welcome state
if st.session_state.rag is None:
    st.info("👈 Enter your Gemini API key, upload PDFs, and click **Process Documents** to begin.")
    with st.expander("💡 What can I ask?"):
        st.markdown("""
- *"Summarise the key findings of this document"*
- *"What are the main risks mentioned?"*
- *"List all recommendations from chapter 3"*
- *"Compare the methodology in document 1 vs document 2"*
- *"What does the author conclude about X?"*
        """)
