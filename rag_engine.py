import os
import tempfile
from typing import List, Dict, Any

import google.generativeai as genai
from langchain.embeddings.base import Embeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate


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


class GeminiEmbeddings(Embeddings):
    """Custom embeddings class using google-generativeai SDK directly."""
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = "models/text-embedding-004"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(result["embedding"])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        result = genai.embed_content(
            model=self.model,
            content=text,
            task_type="retrieval_query"
        )
        return result["embedding"]


class RAGEngine:
    def __init__(self, api_key: str, chunk_size: int = 500, top_k: int = 4):
        self.api_key = api_key
        self.chunk_size = chunk_size
        self.top_k = top_k
        self.vectordb = None
        self.chain = None

        os.environ["GOOGLE_API_KEY"] = api_key
        genai.configure(api_key=api_key)

        self.embeddings = GeminiEmbeddings(api_key=api_key)

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.2,
            convert_system_message_to_human=True
        )

    def _load_pdfs(self, uploaded_files):
        all_docs = []
        file_meta = {}
        for uf in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uf.read())
                tmp_path = tmp.name
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = uf.name
            all_docs.extend(docs)
            file_meta[uf.name] = len(docs)
            os.unlink(tmp_path)
        return all_docs, file_meta

    def _chunk_documents(self, docs):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=int(self.chunk_size * 0.2),
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
        return splitter.split_documents(docs)

    def _build_vectorstore(self, chunks):
        return Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name="docmind_collection"
        )

    def _build_chain(self, vectordb
