import os
import re
import json
import html
import uuid

from functools import lru_cache
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

from sentence_transformers import CrossEncoder


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

USER_AGENT = os.getenv(
    "USER_AGENT",
    "Mozilla/5.0 (RealEstateRAG/2.0)"
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

LLM_MODEL = "openai/gpt-oss-20b"

CHUNK_SIZE = 150
CHUNK_OVERLAP = 25


# ============================================================
# MODELS
# ============================================================

@lru_cache(maxsize=1)
def get_embedding_model():

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


@lru_cache(maxsize=1)
def get_reranker():

    return CrossEncoder(
        RERANK_MODEL
    )


@lru_cache(maxsize=1)
def get_llm():

    if not GROQ_API_KEY:
        raise EnvironmentError(
            "GROQ_API_KEY is missing."
        )

    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=LLM_MODEL,
        temperature=0,
        max_tokens=1200
    )


# ============================================================
# JSON-LD HELPERS
# ============================================================

def iter_json_objects(value):

    if isinstance(value, dict):

        yield value

        for child in value.values():
            yield from iter_json_objects(child)

    elif isinstance(value, list):

        for child in value:
            yield from iter_json_objects(child)


def extract_article_schema(soup):

    scripts = soup.find_all(
        "script",
        type="application/ld+json"
    )

    fallback = None

    article_types = {
        "Article",
        "NewsArticle",
        "ReportageNewsArticle"
    }

    for script in scripts:

        raw = script.string or script.get_text()

        if not raw:
            continue

        try:
            data = json.loads(raw)

        except Exception:
            continue

        for obj in iter_json_objects(data):

            obj_type = obj.get("@type")

            if isinstance(obj_type, list):
                types = set(obj_type)

            else:
                types = {obj_type}

            if obj.get("articleBody"):

                if types.intersection(article_types):
                    return obj

                if fallback is None:
                    fallback = obj

    return fallback


def extract_author(schema):

    if not schema:
        return ""

    author = schema.get("author")

    if isinstance(author, dict):
        return author.get("name", "")

    if isinstance(author, list):

        names = []

        for item in author:

            if isinstance(item, dict):
                name = item.get("name")

                if name:
                    names.append(name)

        return ", ".join(names)

    return ""


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if not text:
        return ""

    # Decode HTML entities
    text = html.unescape(
        html.unescape(text)
    )

    text = text.replace(
        "\xa0",
        " "
    )

    lines = []

    skip_exact = {
        "twitter",
        "facebook",
        "whatsapp",
        "reddit",
        "email",
        "advertisement",
        "share"
    }

    skip_prefixes = (
        "also read",
        "read more",
        "advertisement"
    )

    previous = None

    for line in text.splitlines():

        line = re.sub(
            r"\s+",
            " ",
            line
        ).strip()

        if not line:
            continue

        lowered = line.lower()

        if lowered in skip_exact:
            continue

        if lowered.startswith(skip_prefixes):
            continue

        # Remove consecutive duplicate lines
        if line == previous:
            continue

        lines.append(line)

        previous = line

    return "\n\n".join(lines)


# ============================================================
# URL LOADER
# ============================================================

def load_url_document(url):

    headers = {
        "User-Agent": USER_AGENT,
        "Accept-Language": "en-US,en;q=0.9"
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=20
    )

    response.raise_for_status()

    # Helps websites such as TOI decode correctly
    if response.apparent_encoding:
        response.encoding = response.apparent_encoding

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    schema = extract_article_schema(
        soup
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    page_title = (
        soup.title.get_text(strip=True)
        if soup.title
        else "Unknown title"
    )

    if schema:

        title = schema.get(
            "headline",
            page_title
        )

        published_date = schema.get(
            "datePublished",
            ""
        )

        author = extract_author(
            schema
        )

    else:

        title = page_title
        published_date = ""
        author = ""

    # --------------------------------------------------------
    # Main article text
    # --------------------------------------------------------

    article_body = (
        schema.get("articleBody")
        if schema
        else None
    )

    if article_body:

        text = clean_text(
            article_body
        )

    else:

        # Remove obvious website noise
        for tag in soup([
            "script",
            "style",
            "noscript",
            "svg",
            "form",
            "button",
            "nav",
            "footer",
            "header",
            "aside"
        ]):
            tag.decompose()

        container = (
            soup.select_one("article")
            or soup.select_one("main")
            or soup.body
        )

        if container is None:
            raise RuntimeError(
                "No readable page content found."
            )

        raw_text = container.get_text(
            separator="\n",
            strip=True
        )

        text = clean_text(
            raw_text
        )

    if len(text) < 100:

        raise RuntimeError(
            "Page did not contain enough readable text."
        )

    return Document(
        page_content=text,
        metadata={
            "source": url,
            "title": title,
            "domain": urlparse(url).netloc,
            "published_date": published_date,
            "author": author
        }
    )


# ============================================================
# MULTI-URL LOADING
# ============================================================

def load_documents(
    urls,
    progress_callback=None
):

    documents = []
    failures = []

    total = len(urls)

    for index, url in enumerate(
        urls,
        start=1
    ):

        try:

            doc = load_url_document(
                url
            )

            documents.append(
                doc
            )

            success = True
            message = "Loaded"

        except Exception as exc:

            failures.append({
                "url": url,
                "error": str(exc)
            })

            success = False
            message = str(exc)

        if progress_callback:

            progress_callback(
                index,
                total,
                url,
                success,
                message
            )

    return documents, failures


# ============================================================
# CHUNKING
# ============================================================

def split_documents(documents):

    splitter = (
        RecursiveCharacterTextSplitter
        .from_tiktoken_encoder(
            encoding_name="cl100k_base",

            chunk_size=CHUNK_SIZE,

            chunk_overlap=CHUNK_OVERLAP,

            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]
        )
    )

    chunks = splitter.split_documents(
        documents
    )

    return chunks


# ============================================================
# BUILD VECTOR DATABASE
# ============================================================

def build_knowledge_base(
    urls,
    progress_callback=None
):

    # Remove duplicate URLs
    urls = list(
        dict.fromkeys(urls)
    )

    documents, failures = load_documents(
        urls,
        progress_callback
    )

    if not documents:

        raise RuntimeError(
            "None of the supplied URLs could be processed."
        )

    chunks = split_documents(
        documents
    )

    embedding_model = get_embedding_model()

    collection_name = (
        "real_estate_"
        + uuid.uuid4().hex[:12]
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=collection_name
    )

    domains = sorted({
        doc.metadata["domain"]
        for doc in documents
    })

    return {
        "vectorstore": vectorstore,
        "documents": documents,
        "chunks": chunks,
        "domains": domains,
        "failures": failures
    }


# ============================================================
# MULTI-QUERY GENERATION
# ============================================================

def generate_search_queries(
    question,
    number_of_queries=3
):

    llm = get_llm()

    prompt = f"""
Rewrite the user question into {number_of_queries}
short semantic-search queries.

Each query should focus on a different aspect
of the original question.

Return ONLY one query per line.
Do not number them.

User question:
{question}
"""

    try:

        response = llm.invoke(
            prompt
        )

        generated = []

        for line in response.content.splitlines():

            line = re.sub(
                r"^\s*(?:[-*]|\d+[.)])\s*",
                "",
                line
            ).strip()

            if line:
                generated.append(line)

        generated = generated[
            :number_of_queries
        ]

    except Exception:

        generated = []

    # Always include original query
    queries = [question] + generated

    # Remove duplicates while preserving order
    return list(
        dict.fromkeys(queries)
    )


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_candidates(
    vectorstore,
    queries,
    allowed_domains=None,
    per_query_k=5,
    use_mmr=False
):

    candidates = {}

    metadata_filter = None

    if allowed_domains:

        metadata_filter = {
            "domain": {
                "$in": allowed_domains
            }
        }

    for query in queries:

        if use_mmr:

            docs = vectorstore.max_marginal_relevance_search(
                query,
                k=per_query_k,
                fetch_k=max(
                    per_query_k * 3,
                    10
                ),
                filter=metadata_filter
            )

        else:

            docs = vectorstore.similarity_search(
                query,
                k=per_query_k,
                filter=metadata_filter
            )

        for doc in docs:

            key = (
                doc.metadata.get(
                    "source",
                    ""
                ),
                doc.page_content
            )

            candidates[key] = doc

    return list(
        candidates.values()
    )


# ============================================================
# RERANKING + SOURCE BALANCING
# ============================================================

def rerank_documents(
    question,
    documents,
    final_k=5,
    max_per_source=2
):

    if not documents:
        return []

    reranker = get_reranker()

    pairs = [
        [
            question,
            doc.page_content
        ]
        for doc in documents
    ]

    scores = reranker.predict(
        pairs
    )

    ranked = sorted(
        zip(documents, scores),
        key=lambda item: item[1],
        reverse=True
    )

    selected = []
    skipped = []

    source_count = {}

    # First pass — source balancing
    for doc, score in ranked:

        domain = doc.metadata.get(
            "domain",
            "unknown"
        )

        if (
            source_count.get(domain, 0)
            < max_per_source
        ):

            selected.append(
                (doc, float(score))
            )

            source_count[domain] = (
                source_count.get(
                    domain,
                    0
                ) + 1
            )

        else:

            skipped.append(
                (doc, float(score))
            )

        if len(selected) == final_k:
            break

    # If there are not enough different sources,
    # fill remaining slots with highest scoring chunks
    if len(selected) < final_k:

        selected_keys = {
            (
                doc.metadata.get(
                    "source"
                ),
                doc.page_content
            )
            for doc, _ in selected
        }

        for doc, score in skipped:

            key = (
                doc.metadata.get(
                    "source"
                ),
                doc.page_content
            )

            if key in selected_keys:
                continue

            selected.append(
                (doc, score)
            )

            if len(selected) == final_k:
                break

    return selected


# ============================================================
# FINAL RAG PIPELINE
# ============================================================

def ask_rag(
    vectorstore,
    question,
    allowed_domains=None,
    use_multi_query=True,
    use_mmr=False,
    final_k=5,
    max_per_source=2
):

    if not question.strip():

        raise ValueError(
            "Question cannot be empty."
        )

    # --------------------------------------------------------
    # 1. Query generation
    # --------------------------------------------------------

    if use_multi_query:

        queries = generate_search_queries(
            question
        )

    else:

        queries = [
            question
        ]

    # --------------------------------------------------------
    # 2. Vector retrieval
    # --------------------------------------------------------

    candidates = retrieve_candidates(
        vectorstore=vectorstore,
        queries=queries,
        allowed_domains=allowed_domains,
        per_query_k=5,
        use_mmr=use_mmr
    )

    if not candidates:

        return {
            "answer": (
                "The provided sources do not "
                "contain enough information."
            ),
            "sources": [],
            "debug": {
                "queries": queries,
                "candidate_count": 0,
                "final_chunks": []
            }
        }

    # --------------------------------------------------------
    # 3. Reranking
    # --------------------------------------------------------

    ranked_docs = rerank_documents(
        question=question,
        documents=candidates,
        final_k=final_k,
        max_per_source=max_per_source
    )

    # --------------------------------------------------------
    # 4. Build citations/context
    # --------------------------------------------------------

    source_ids = {}
    source_info = {}
    context_parts = []

    for doc, score in ranked_docs:

        url = doc.metadata.get(
            "source",
            ""
        )

        title = doc.metadata.get(
            "title",
            "Unknown title"
        )

        domain = doc.metadata.get(
            "domain",
            ""
        )

        if url not in source_ids:

            source_number = (
                len(source_ids) + 1
            )

            source_ids[url] = (
                source_number
            )

            source_info[
                source_number
            ] = {
                "title": title,
                "url": url,
                "domain": domain
            }

        source_number = (
            source_ids[url]
        )

        context_parts.append(
            f"[Source {source_number}]\n"
            f"Title: {title}\n"
            f"{doc.page_content}"
        )

    context = "\n\n".join(
        context_parts
    )

    # --------------------------------------------------------
    # 5. LLM generation
    # --------------------------------------------------------

    llm = get_llm()

    prompt = f"""
You are a source-grounded real-estate
research assistant.

Answer the question using ONLY the provided
context.

Rules:
- Do not use outside knowledge.
- Every factual point must cite its supporting
  source using [Source X].
- Preserve numbers, dates, quarters and time
  periods exactly.
- Never treat "year-to-date", "corresponding
  period", and "full year" as the same period.
- If sources disagree, clearly mention the
  disagreement.
- Do not invent facts or citations.
- Keep the answer clear and concise.
- If the context is insufficient, say:
  "The provided sources do not contain enough information."

Question:
{question}

Context:
{context}

Answer:
"""

    response = llm.invoke(
        prompt
    )

    # --------------------------------------------------------
    # 6. Final source list
    # --------------------------------------------------------

    # Only return sources actually cited in the answer
    cited_numbers = {
        int(num)
        for num in re.findall(
            r"\[Source\s+(\d+)\]",
            response.content
        )
    }

    sources = []

    for number, info in source_info.items():

        if number in cited_numbers:
            sources.append({
                "number": number,
                "title": info["title"],
                "url": info["url"],
                "domain": info["domain"]
            })

    # --------------------------------------------------------
    # Debug information
    # --------------------------------------------------------

    debug_chunks = []

    for doc, score in ranked_docs:

        debug_chunks.append({
            "domain": doc.metadata.get(
                "domain",
                ""
            ),
            "rerank_score": round(
                float(score),
                4
            ),
            "preview": (
                doc.page_content[:400]
            )
        })

    return {
        "answer": response.content,
        "sources": sources,
        "debug": {
            "queries": queries,
            "candidate_count": len(
                candidates
            ),
            "final_chunks": debug_chunks
        }
    }