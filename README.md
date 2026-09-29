# 🏠 Real Estate RAG Assistant

An end-to-end **Retrieval-Augmented Generation (RAG)** application for researching the Indian real-estate market using multiple live web sources.

The system ingests real-estate articles and market reports, cleans and chunks the content, converts it into embeddings, stores it in **ChromaDB**, performs advanced retrieval and reranking, and generates **source-grounded answers with citations** using an LLM through Groq.

> The project was built not only as an application, but also as a hands-on exploration of how modern RAG systems work internally — from chunking and embeddings to reranking, source balancing, grounding, and evaluation.


## 🌐 Live Demo

[🚀 Launch Real Estate RAG Assistant](https://real-estate-rag-assistant-p.streamlit.app/)

---

## 🚀 What This Project Does

Users can provide multiple real-estate research URLs and ask natural-language questions such as:

- What were the major trends in India's real-estate market in Q3 2026?
- How did housing launches change in Pune?
- Which cities saw housing sales increase?
- How much institutional capital entered Indian real estate?
- What is the expected size of India's real-estate market by 2031?
- What do different research sources disagree on?

The assistant retrieves relevant evidence from the supplied sources and answers using **only that evidence**.

If sufficient information is not available, the system responds:

> **The provided sources do not contain enough information.**

This prevents the application from simply using unsupported general knowledge.

---

## ✨ Key Features

### 📥 Multi-Source Web Ingestion

- Accepts multiple URLs
- Loads live web pages
- Extracts article content
- Uses JSON-LD metadata where available
- Extracts:
  - title
  - author
  - publication date
  - source URL
  - domain

---

### 🧹 Content Cleaning

The ingestion pipeline removes unnecessary webpage content such as:

- navigation elements
- scripts
- styles
- advertisements
- social sharing labels
- duplicate lines
- unrelated webpage noise

This improves the quality of downstream embeddings and retrieval.

---

### ✂️ Token-Based Chunking

Documents are split using:

`RecursiveCharacterTextSplitter`

with token-aware sizing through `tiktoken`.

Current configuration:

```python
CHUNK_SIZE = 150
CHUNK_OVERLAP = 25
```

This was selected after experimenting with:

- character-based chunking
- recursive chunking
- HTML-aware chunking
- semantic chunking
- different chunk sizes
- different overlap sizes

---

### 🧠 Semantic Embeddings

Embedding model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The embedding model converts document chunks and user questions into numerical vectors representing semantic meaning.

---

### 🗄️ ChromaDB Vector Store

All chunk embeddings are stored inside **ChromaDB**.

This enables semantic retrieval based on meaning rather than exact keyword matching.

---

### 🔎 Multi-Query Retrieval

Broad questions can be difficult for a single embedding search.

The system therefore generates multiple semantic search queries from the original question.

Example:

```text
Original Question:
What were the major trends in India's real estate market in Q3 2026?

Generated Queries:
- India real estate Q3 2026 trends
- Q3 2026 residential property trends
- Q3 2026 commercial real estate trends
```

Results from multiple searches are combined and deduplicated before reranking.

---

### 🎯 Similarity Search & MMR

The application supports two retrieval strategies:

#### Similarity Search

Retrieves chunks closest to the user's question in embedding space.

#### Maximum Marginal Relevance — MMR

Attempts to balance:

```text
Relevance + Diversity
```

This can reduce duplicated information in retrieved context.

Users can switch between both approaches from the Streamlit interface.

---

### 🏷️ Metadata Filtering

Retrieval can be restricted using document metadata.

The current application supports domain-level filtering, allowing users to decide which sources should participate in retrieval.

Example:

```text
Business Standard
Times of India
NDTV Profit
Mordor Intelligence
MarkNtel Advisors
```

This follows an important RAG principle:

```text
Structured filtering first
        ↓
Semantic retrieval second
```

---

### 🧮 Cross-Encoder Reranking

Vector search is fast, but the most semantically similar chunk is not always the chunk containing the best answer.

Retrieved candidates are therefore reranked using:

```text
cross-encoder/ms-marco-MiniLM-L6-v2
```

Pipeline:

```text
Vector Search
     ↓
Candidate Chunks
     ↓
Cross-Encoder
     ↓
Better Relevance Ranking
```

The CrossEncoder evaluates the **question and document together**, providing a stronger relevance score than embedding similarity alone.

---

### ⚖️ Source Balancing

Large reports can contain dozens of chunks and dominate retrieval results.

To prevent one source from taking over the entire context window, the application limits the number of final chunks selected from the same source.

Example:

```text
Max chunks per source = 2
```

This encourages the final context to contain evidence from multiple independent sources.

---

### 📚 Grounded Answer Generation

The final prompt explicitly instructs the LLM to:

- answer only from retrieved context
- avoid outside knowledge
- preserve numbers and dates
- preserve time periods correctly
- cite factual claims
- acknowledge conflicting sources
- avoid unsupported conclusions

The application uses:

```text
openai/gpt-oss-20b
```

through the **Groq API**.

Temperature:

```text
0
```

This keeps responses more deterministic and appropriate for factual research.

---

### 🔗 Source Citations

Answers use citations such as:

```text
[Source 1]
[Source 2]
[Source 3]
```

Source URLs are controlled by Python rather than generated by the LLM.

Only sources actually cited in the final answer are displayed to the user.

---

### 🛡️ Insufficient-Context Guardrail

If retrieved evidence does not support the question, the system does not guess.

Example:

```text
Question:
Which real-estate stock should I buy?

Answer:
The provided sources do not contain enough information.
```

Even if some retrieved chunks mention company names, the model will not transform simple company mentions into unsupported investment recommendations.

---

### 🔍 Retrieval Debug Panel

The Streamlit application includes a debugging section showing:

- generated search queries
- total candidate chunks
- reranking scores
- source domains
- final context chunks

This makes the internal RAG pipeline visible instead of treating retrieval as a black box.

---

# 🏗️ RAG Architecture

```text
                     ┌───────────────────────┐
                     │      Source URLs      │
                     └───────────┬───────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │    Web Extraction     │
                     │ BeautifulSoup/JSON-LD │
                     └───────────┬───────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │   Content Cleaning    │
                     │      + Metadata       │
                     └───────────┬───────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │ Token-Based Chunking  │
                     │   150 / overlap 25    │
                     └───────────┬───────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │ HuggingFace Embedding Model  │
                  │      all-MiniLM-L6-v2        │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                       ┌─────────────────┐
                       │    ChromaDB     │
                       │  Vector Store   │
                       └────────┬────────┘
                                │
                         User Question
                                │
                                ▼
                    ┌─────────────────────┐
                    │ Multi-Query Rewrite │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Semantic Retrieval  │
                    │ Similarity / MMR    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Metadata Filtering  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Deduplication    │
                    └──────────┬──────────┘
                               │
                               ▼
                 ┌────────────────────────────┐
                 │ Cross-Encoder Reranking    │
                 └─────────────┬──────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Source Balancing   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Context Construction│
                    │ + Citation Mapping  │
                    └──────────┬──────────┘
                               │
                               ▼
                       ┌───────────────┐
                       │   Groq LLM    │
                       │ GPT-OSS-20B   │
                       └───────┬───────┘
                               │
                               ▼
                 ┌─────────────────────────┐
                 │ Grounded Answer         │
                 │ + Verified Source Links │
                 └─────────────────────────┘
```

---

# 🧰 Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| Frontend | Streamlit |
| LLM Framework | LangChain |
| LLM Provider | Groq |
| LLM | GPT-OSS-20B |
| Embeddings | Hugging Face Sentence Transformers |
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Database | ChromaDB |
| Reranking | Sentence Transformers CrossEncoder |
| Reranker | ms-marco-MiniLM-L6-v2 |
| Web Scraping | Requests + BeautifulSoup |
| Tokenization | tiktoken |
| Environment | Python Virtual Environment |

---

# 📊 RAG Evaluation

The system was evaluated using a small manually created question-answer test set.

## Retrieval Evaluation

### Recall@3

Recall@K answers:

> Did the correct evidence appear somewhere within the Top-K retrieved chunks?

Result:

```text
Recall@3 = 1.00
```

All test questions had relevant evidence within the Top 3 retrieved chunks.

---

### Mean Reciprocal Rank — MRR

MRR measures how highly the first correct result appears.

Result:

```text
MRR = 0.875
```

This showed that relevant evidence usually appeared very high in the retrieval ranking.

---

## End-to-End RAG Evaluation

The complete pipeline was tested as:

```text
Question
   ↓
Retrieval
   ↓
Context
   ↓
LLM
   ↓
Answer
   ↓
Expected Fact Comparison
```

Test results:

```text
4 / 4 questions passed
```

Example questions included:

```text
How did Pune housing launches change in Q3 2026?

How much capital entered India's real estate market in Q3 2026?

Which cities saw housing sales increase in Q3 2026?

What is the expected size of India's real estate market by 2031?
```

> These metrics come from a small learning-oriented evaluation set and should not be interpreted as production benchmark results.

---

# 🧪 Important Experiments & Findings

Several retrieval strategies were tested before building the final pipeline.

### Character vs Recursive Chunking

Recursive splitting proved more reliable because it can progressively fall back from:

```text
Paragraph
→ Newline
→ Sentence
→ Space
→ Character
```

instead of producing oversized chunks.

---

### HTML-Aware Chunking

HTML header-based chunking was tested but performed poorly on noisy news pages where webpage structure did not represent meaningful document sections.

For this project:

```text
Clean article text + recursive token chunking
```

performed better.

---

### Semantic Chunking

Sentence-embedding similarity was tested for semantic boundary detection.

However, it created uneven and sometimes fragmented chunks on short news articles.

The final application therefore uses token-aware recursive chunking.

---

### Vector Similarity vs Answer Relevance

An important observation was:

> The most semantically similar chunk is not necessarily the chunk containing the best answer.

This motivated adding a CrossEncoder reranking stage.

---

### MMR

MMR was tested to introduce diversity into retrieved results.

It becomes more useful when the knowledge base contains many similar or duplicated chunks.

---

### Multi-Query Retrieval

Multi-query retrieval improved coverage for broad analytical questions, although it did not automatically guarantee source diversity.

This led to the additional **source-balancing** stage.

---

### Citation ≠ Guaranteed Faithfulness

During experimentation, the model once had the correct evidence and citation but misinterpreted a numerical comparison between:

```text
corresponding period
```

and:

```text
full-year total
```

This demonstrated an important lesson:

> Correct retrieval and citations do not automatically guarantee correct reasoning.

The final prompt therefore explicitly instructs the model to preserve dates, quarters, comparison periods, and numerical relationships.

---

# 📁 Project Structure

```text
real-estate-rag-assistant/
│
├── notebooks/
│   └── 01_rag_fundamentals.ipynb
│
├── src/
│   ├── __init__.py
│   └── rag.py
│
├── .env
├── .env.example
├── .gitignore
├── app.py
├── README.md
└── requirements.txt
```

### Important

The local `.venv/` directory and `.env` file are intentionally excluded from GitHub using `.gitignore`.

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/Krushang010/real-estate-rag-assistant.git
```

```bash
cd real-estate-rag-assistant
```

---

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

---

## 4. Configure Environment Variables

Create a `.env` file in the project root.

```env
GROQ_API_KEY=your_groq_api_key_here
USER_AGENT=Mozilla/5.0 (RealEstateRAG/2.0)
```

Do **not** commit the `.env` file to GitHub.

An example configuration is provided in:

```text
.env.example
```

---

# ▶️ Run the Application

Start Streamlit:

```bash
python -m streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

# 💻 Application Workflow

### Step 1 — Add Sources

Paste one URL per line into the sidebar.

Example source types:

```text
News articles
Market research reports
Industry analysis
Real-estate research publications
```

---

### Step 2 — Build Knowledge Base

Click:

```text
Process Sources
```

The system will:

```text
Download pages
→ Clean content
→ Extract metadata
→ Split documents
→ Generate embeddings
→ Store vectors in ChromaDB
```

---

### Step 3 — Configure Retrieval

Users can control:

- source/domain selection
- multi-query retrieval
- similarity vs MMR
- number of final context chunks
- maximum chunks per source

---

### Step 4 — Ask Questions

Example:

```text
What were the major trends in India's real-estate market in Q3 2026?
```

The application retrieves, reranks, synthesizes, and cites supporting evidence.

---

# 🧠 What I Learned

This project helped me understand RAG beyond simply calling a framework abstraction.

Concepts explored include:

- document ingestion
- webpage cleaning
- structured metadata
- JSON-LD extraction
- character vs token chunking
- chunk size
- chunk overlap
- recursive chunking
- HTML-aware chunking
- semantic chunking
- tokenization
- embeddings
- cosine similarity
- vector databases
- ChromaDB
- Top-K retrieval
- semantic similarity
- MMR
- metadata filtering
- multi-query retrieval
- retrieval deduplication
- CrossEncoder reranking
- source balancing
- context construction
- grounded generation
- citation handling
- hallucination control
- retrieval evaluation
- Recall@K
- Mean Reciprocal Rank
- answer evaluation
- numerical reasoning limitations in LLMs

---

# ⚠️ Limitations

The current implementation still has several limitations:

- Some websites block automated scraping.
- Page layouts can change and break extraction.
- Retrieval quality depends heavily on source quality.
- Very large knowledge bases would require more scalable infrastructure.
- Current Chroma collections are session-oriented rather than a permanent production database.
- Multi-query generation requires an LLM call.
- CrossEncoder reranking increases latency.
- The evaluation set is intentionally small.
- Correct retrieved evidence does not guarantee perfect LLM reasoning.
- The application should not be treated as financial or investment advice.

---

# 🔮 Possible Future Improvements

Future versions could include:

- persistent Chroma collections
- Pinecone / Qdrant / Weaviate integration
- hybrid BM25 + semantic search
- richer metadata extraction
- city and date filters
- query routing
- parent-document retrieval
- contextual compression
- stronger reranking models
- automated RAG evaluation
- RAGAS / DeepEval style evaluation
- claim-level citation verification
- deterministic numerical verification
- PDF support
- conversation memory
- authentication
- API backend
- Docker deployment
- cloud deployment
- observability and tracing

---

# 🎯 Project Objective

The main objective of this project was not simply to create a chatbot.

It was to understand and implement the complete retrieval process:

```text
How should documents be cleaned?

How should they be chunked?

How are embeddings created?

How does vector retrieval work?

Why does Top-K sometimes fail?

When should metadata filtering be used?

Why is reranking useful?

How can multiple sources be balanced?

How do we prevent unsupported answers?

How should a RAG system be evaluated?
```

The final Streamlit application is the production-style outcome of those experiments.

---

# 👤 Author

## Krushang Patel

**Data Scientist**

Focused on building end-to-end machine-learning and AI systems across:

- Machine Learning
- Forecasting
- NLP
- Retrieval-Augmented Generation
- Generative AI
- Decision Intelligence

GitHub: [Krushang010](https://github.com/Krushang010)

LinkedIn: [Krushang Patel](https://www.linkedin.com/in/krushang-patel-data-scientist)

---

## ⭐ If you found this project useful

Consider giving the repository a **star**.

It helps support the project and future improvements.