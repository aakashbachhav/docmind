"""
rag_engine.py — Core RAG pipeline for DocMind

Pipeline:
  PDF Upload → Text Extraction → Chunking → Embedding → ChromaDB Storage
  Query → Query Embedding → Similarity Search → Context Assembly → Gemini LLM → Answer + Sources
"""

import os
import tempfile
from typing import List, Dict, Any

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate


# ── Custom RAG prompt ─────────────────────────────────────────────────────────
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template="""You are DocMind, an intelligent document assistant. Answer the user's question
using ONLY the provided context from their uploaded documents. Be precise and cite which
part of the document supports your answer. If the answer is not in the context, say clearly:
"I couldn't find information about this in the uploaded documents."

Context from documents:
{context}

Chat history:
{chat_history}

User question: {question}

Answer (be thorough but concise, use bullet points for lists):"""
)


class RAGEngine:
    """
    End-to-end Retrieval-Augmented Generation pipeline.

    Attributes:
        api_key   : Gemini API key
        chunk_size: Token size for each document chunk
        top_k     : Number of chunks to retrieve per query
        vectordb  : ChromaDB vector store instance
        chain     : LangChain ConversationalRetrievalChain
    """

    def __init__(self, api_key: str, chunk_size: int = 500, top_k: int = 4):
        self.api_key = api_key
        self.chunk_size = chunk_size
        self.top_k = top_k
        self.vectordb = None
        self.chain = None

        os.environ["GOOGLE_API_KEY"] = api_key

        # Embedding model — converts text to dense vectors
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=api_key
        )

        # LLM — generates the final answer
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.2,        # Low temperature = more factual, less creative
            convert_system_message_to_human=True
        )

    # ── Step 1: Load & parse PDFs ─────────────────────────────────────────────
    def _load_pdfs(self, uploaded_files) -> tuple[list, dict]:
        """Save uploaded Streamlit files to temp disk, load with PyPDFLoader."""
        all_docs = []
        file_meta = {}

        for uf in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uf.read())
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

            # Attach filename metadata to each page
            for doc in docs:
                doc.metadata["source"] = uf.name
            all_docs.extend(docs)
            file_meta[uf.name] = len(docs)
            os.unlink(tmp_path)

        return all_docs, file_meta

    # ── Step 2: Chunk documents ───────────────────────────────────────────────
    def _chunk_documents(self, docs: list) -> list:
        """
        Split documents into overlapping chunks.
        chunk_overlap ensures context isn't lost at chunk boundaries.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=int(self.chunk_size * 0.2),   # 20% overlap
            separators=["\n\n", "\n", ". ", " ", ""],   # Priority order
            length_function=len
        )
        return splitter.split_documents(docs)

    # ── Step 3: Embed & store in ChromaDB ────────────────────────────────────
    def _build_vectorstore(self, chunks: list) -> Chroma:
        """
        Embed all chunks using Gemini embeddings and store in ChromaDB.
        ChromaDB runs in-memory here (no server needed).
        """
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name="docmind_collection"
        )
        return vectordb

    # ── Step 4: Build retrieval chain ─────────────────────────────────────────
    def _build_chain(self, vectordb: Chroma) -> ConversationalRetrievalChain:
        """
        Wire together: retriever → memory → LLM → answer
        ConversationBufferWindowMemory keeps last 5 exchanges for context.
        """
        retriever = vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k}
        )
        memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            output_key="answer",
            return_messages=True,
            k=5  # Remember last 5 Q&A pairs
        )
        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": RAG_PROMPT},
            return_source_documents=True,
            verbose=False
        )
        return chain

    # ── Public: Process all uploaded files ───────────────────────────────────
    def load_documents(self, uploaded_files) -> Dict[str, Any]:
        """Full pipeline: load → chunk → embed → index. Returns stats."""
        docs, file_meta = self._load_pdfs(uploaded_files)
        chunks = self._chunk_documents(docs)
        self.vectordb = self._build_vectorstore(chunks)
        self.chain = self._build_chain(self.vectordb)

        return {
            "num_docs": len(uploaded_files),
            "total_pages": len(docs),
            "total_chunks": len(chunks),
            "files": file_meta
        }

    # ── Public: Query the RAG pipeline ───────────────────────────────────────
    def query(self, question: str) -> Dict[str, Any]:
        """
        Run a query through the full RAG pipeline.
        Returns answer text + source chunk metadata.
        """
        if not self.chain:
            raise RuntimeError("No documents loaded. Call load_documents() first.")

        result = self.chain.invoke({"question": question})

        # Extract source metadata for citations
        sources = []
        seen = set()
        for doc in result.get("source_documents", []):
            key = (doc.metadata.get("source", ""), doc.metadata.get("page", 0))
            if key not in seen:
                seen.add(key)
                sources.append({
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", 0) + 1,  # 0-indexed → 1-indexed
                    "snippet": doc.page_content[:200].strip() + "..."
                })

        return {
            "answer": result["answer"],
            "sources": sources
        }
