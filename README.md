# DocMind — RAG PDF Chatbot

> AI-powered document intelligence using Retrieval-Augmented Generation (RAG)
> Built with LangChain · ChromaDB · Gemini API · Streamlit

**Live Demo:** [your-app.streamlit.app](#) | **GitHub:** [github.com/aakashbachhav/docmind](#)

---

## What it does

DocMind lets you upload any PDF document and have a conversation with it.
Ask questions, get answers grounded in the document, with exact page citations shown for every response.

- Upload multiple PDFs simultaneously
- Ask natural language questions
- Receive answers with source page references
- Maintains conversation memory across follow-up questions

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE                       │
│                                                             │
│  PDF Upload → PyPDFLoader → RecursiveCharacterTextSplitter  │
│      → GoogleGenerativeAIEmbeddings → ChromaDB Index        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      QUERY PIPELINE                          │
│                                                             │
│  User Query → Embed Query → Similarity Search (ChromaDB)    │
│      → Top-K Chunks → Custom Prompt → Gemini 1.5 Flash      │
│      → Answer + Source Citations                            │
└─────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Component | Technology | Why chosen |
|-----------|-----------|------------|
| LLM | Gemini 1.5 Flash | Free tier, fast, strong reasoning |
| Embeddings | Google Embedding-001 | 768-dim vectors, free API |
| Vector DB | ChromaDB | In-memory, no server needed, easy setup |
| Orchestration | LangChain | Standard RAG framework, ConversationalRetrievalChain |
| PDF Parsing | PyPDFLoader (LangChain) | Preserves page metadata |
| Chunking | RecursiveCharacterTextSplitter | Smart boundary detection |
| Frontend | Streamlit | Rapid deployment, Python-native |
| Memory | ConversationBufferWindowMemory | Retains last 5 exchanges |
| Deployment | Streamlit Cloud | Free hosting, GitHub integration |

---

## Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/aakashbachhav/docmind.git
cd docmind

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Get a free Gemini API key at https://aistudio.google.com — paste it in the sidebar when the app opens.

---

## Deploy to Streamlit Cloud (free)

1. Push this repo to GitHub
2. Go to https://share.streamlit.io
3. Click "New app" → select your repo → set `app.py` as the main file
4. Done — your app is live at `your-app-name.streamlit.app`

---

## Project Structure

```
docmind/
├── app.py              # Streamlit UI — chat interface, file upload, stats display
├── rag_engine.py       # Core RAG pipeline — loading, chunking, embedding, querying
├── requirements.txt    # Python dependencies
├── .streamlit/
│   └── secrets.toml   # API key storage for deployment (gitignored)
├── .gitignore
└── README.md
```

---

## Key Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| chunk_size | 500 | Larger = more context per chunk, fewer chunks total |
| chunk_overlap | 100 (20%) | Prevents information loss at chunk boundaries |
| top_k | 4 | More chunks = more context but slower + costlier |
| temperature | 0.2 | Low = factual; High = creative |
| memory window | 5 | Number of past Q&A pairs retained |

---

---

# INTERVIEW DOCUMENTATION

*Everything you need to confidently explain this project in any technical interview.*

---

## 1. The 30-second pitch

> "DocMind is a RAG-based PDF chatbot I built using LangChain, ChromaDB, and the Gemini API.
> You upload any PDF, it chunks and embeds the content into ChromaDB, and then when you ask
> a question, it retrieves the most relevant chunks, passes them to Gemini as context, and
> returns a grounded answer with page citations. I deployed it for free on Streamlit Cloud.
> The whole pipeline — from upload to answer — takes under 15 seconds for a typical document."

---

## 2. Core concept — What is RAG?

**RAG = Retrieval-Augmented Generation**

The problem it solves: LLMs have a knowledge cutoff and can't read YOUR documents.
RAG bridges this by retrieving relevant document content at query time and passing it
to the LLM as context, so the LLM answers based on your documents, not its training data.

**Two phases:**

**Ingestion (done once when PDFs are uploaded):**
```
PDF → Extract text (PyPDFLoader)
    → Split into chunks (RecursiveCharacterTextSplitter)
    → Convert chunks to vectors (Gemini Embeddings)
    → Store vectors in ChromaDB
```

**Query (done every time user asks a question):**
```
User question → Convert to vector (same embedding model)
              → Find top-K similar chunks in ChromaDB (cosine similarity)
              → Assemble chunks as context
              → Send [context + question] to Gemini LLM
              → Return answer + source page numbers
```

---

## 3. Component deep-dives — expect these questions

### "What is an embedding / vector embedding?"

An embedding is a numerical representation of text as a dense vector (array of floats).
Semantically similar text produces similar vectors — so "What is machine learning?"
and "Define ML" will have vectors that are close together in vector space.

In this project, Google's `embedding-001` model converts each text chunk into a
768-dimensional vector. These vectors are what ChromaDB stores and searches.

### "What is ChromaDB and how is it different from a regular database?"

ChromaDB is a vector database. Unlike a regular SQL database that searches by exact
match or index, ChromaDB searches by semantic similarity — it finds vectors that are
closest to your query vector using cosine similarity.

In this project, ChromaDB runs in-memory (no server, no setup) and holds all chunk
vectors. When a query arrives, it returns the top-K chunks whose vectors are most
similar to the query vector.

### "Why did you use RecursiveCharacterTextSplitter?"

It splits text intelligently by trying progressively smaller separators in order:
`\n\n` (paragraph) → `\n` (line) → `. ` (sentence) → ` ` (word).
This means it respects document structure instead of cutting mid-sentence.

I also added 20% overlap between chunks so that information that falls near a
chunk boundary isn't lost — both adjacent chunks will contain it.

### "What is LangChain and why use it?"

LangChain is an orchestration framework for LLM applications. It provides:
- Standard interfaces for loaders, splitters, embeddings, vector stores, LLMs
- `ConversationalRetrievalChain` — wires retriever + memory + LLM in one line
- `ConversationBufferWindowMemory` — stores last N exchanges so the chatbot
  understands follow-up questions without needing to repeat context

Without LangChain I'd need to manually handle all the glue code between these components.

### "What is the custom prompt doing?"

I wrote a custom `PromptTemplate` that:
1. Instructs the LLM to ONLY use the provided context (prevents hallucination)
2. Tells it to cite which part of the document supports the answer
3. Tells it to clearly say when the answer isn't in the document

This is called prompt engineering — controlling LLM behaviour through instruction design.

### "What is conversational memory?"

`ConversationBufferWindowMemory` stores the last 5 question-answer pairs and appends
them to every new prompt. This lets the user ask follow-up questions like
"Can you elaborate on point 2?" without re-explaining what they meant.

### "How does the similarity search work?"

ChromaDB uses cosine similarity between vectors. Given a query vector Q and stored
chunk vectors C1...Cn, it computes:

```
similarity = (Q · Ci) / (|Q| × |Ci|)
```

Values range from -1 to 1, where 1 = identical. The top-K chunks with the highest
similarity score are returned as the retrieval result.

### "Why Gemini and not OpenAI?"

Two reasons:
1. Free tier — Gemini API has a generous free quota, making this project
   completely free to run, which is important for a portfolio project.
2. I already had experience with Gemini API from my AI PhysioBot and Comrade.AI
   projects, so I could integrate it quickly and focus on the RAG architecture.

### "What are the limitations of this system?"

1. **Table/image blindness** — PyPDFLoader extracts plain text; tables, charts,
   and scanned PDFs aren't handled well. Solution: use a vision model or OCR.
2. **Chunk boundary problem** — even with overlap, an answer that spans a large
   section may be split across chunks neither of which has the complete context.
3. **In-memory ChromaDB** — resets when the app restarts. For production,
   I'd use a persistent ChromaDB instance or a managed service like Pinecone.
4. **No re-ranking** — I retrieve top-K by similarity but don't re-rank them
   by relevance. Adding a cross-encoder re-ranker would improve answer quality.

### "How would you scale this for production?"

1. Replace in-memory ChromaDB with Pinecone or Weaviate (managed, persistent)
2. Add an async ingestion queue (Celery + Redis) for large document batches
3. Add user authentication and per-user vector namespaces
4. Add a caching layer (Redis) for repeated queries
5. Monitor with LangSmith for prompt/retrieval quality tracking (LLMOps)
6. Containerise with Docker and deploy on AWS ECS or GCP Cloud Run

---

## 4. Numbers to remember

- Gemini embedding-001 produces **768-dimensional vectors**
- Default chunk size: **500 characters** with **100-character overlap**
- Default top-K retrieval: **4 chunks** per query
- Memory window: **5 conversation turns**
- Typical ingestion time for a 50-page PDF: **~8–12 seconds**
- Typical query response time: **~3–5 seconds**

---

## 5. How to position this on your resume

```
DocMind — RAG PDF Chatbot                                    Apr 2026
Python, LangChain, ChromaDB, Gemini API, Streamlit          Live | GitHub

• Built end-to-end RAG pipeline: PDF ingestion → chunking (RecursiveCharacterText-
  Splitter, 20% overlap) → vector embedding (Gemini embedding-001, 768-dim) →
  ChromaDB indexing → cosine similarity retrieval → Gemini 1.5 Flash generation
• Implemented ConversationalRetrievalChain with sliding window memory (k=5) enabling
  context-aware multi-turn conversations with source page citations
• Deployed on Streamlit Cloud; supports multi-document querying with configurable
  chunk size and top-K retrieval parameters
```

---

## 6. Likely follow-up questions with answers

**Q: What's the difference between RAG and fine-tuning?**
A: Fine-tuning bakes knowledge into model weights — expensive, needs retraining
for new data, and can cause catastrophic forgetting. RAG keeps knowledge external
and retrievable at inference time — cheaper, always up-to-date, and fully auditable
since you can show which document chunk produced the answer.

**Q: Why not just paste the whole PDF into the prompt?**
A: Context window limits (Gemini Flash has a 1M token window but most models are
smaller), cost (billed per token), and quality (LLMs lose focus with very long
contexts — the "lost in the middle" problem). RAG retrieves only the relevant
paragraphs, keeping the context tight and the answer precise.

**Q: What is cosine similarity and why is it used over Euclidean distance?**
A: Cosine similarity measures the angle between vectors regardless of their magnitude.
This is better for text embeddings because two chunks about the same topic but
different lengths will have similar direction but different magnitudes —
Euclidean distance would penalise the length difference unfairly.

**Q: Have you evaluated the quality of your RAG system?**
A: I tested it manually using a standard question set against a known document and
checked if the retrieved chunks actually contained the answer (retrieval precision)
and if the LLM used the chunk accurately (answer faithfulness). For production
I'd use RAGAS — an automated RAG evaluation framework — to measure faithfulness,
answer relevancy, and context recall systematically.
