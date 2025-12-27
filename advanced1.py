import os
import json
from typing import List, Dict

import numpy as np
import requests
from dotenv import load_dotenv
from pypdf import PdfReader

from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import chromadb
from chromadb.config import Settings

load_dotenv()


# Embedding model
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")


def embed_texts(texts: List[str]) -> np.ndarray:
    return embedder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )


def embed_query(query: str) -> np.ndarray:
    return embed_texts([query])[0]


# OpenRouter LLM
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json"
}

DEFAULT_MODEL = "mistralai/mistral-7b-instruct"


def call_llm(prompt: str) -> str:
    payload = {
        "model": DEFAULT_MODEL,
        "messages": [
            {"role": "system", "content": "Answer strictly using the provided context."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }

    response = requests.post(
        OPENROUTER_URL,
        headers=HEADERS,
        data=json.dumps(payload)
    )

    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# Query classification and normalization
def query_classifier(query: str, call_llm) -> str:
    q = query.lower()

    if any(sig in q for sig in ["0x", "404", "segfault"]):
        return "KEYWORD"

    if any(sig in q for sig in ["compare", "difference", "vs", "versus"]):
        return "DECOMPOSE"

    if q.count(" and ") + q.count(",") >= 2:
        return "DECOMPOSE"

    if len(q.split()) > 14:
        return "DECOMPOSE"

    prompt = f"""
        You are a classifier.

        Return ONLY one of the following tokens:
        KEYWORD
        SEMANTIC
        DECOMPOSE

        No explanations.
        No punctuation.
        No extra text.

        If not sure, return SEMANTIC.

        Query:
        {query}

    """
    return call_llm(prompt).strip().upper()

def normalize_route(text: str) -> str:
    text = text.strip().upper()

    if "KEYWORD" in text:
        return "KEYWORD"
    if "DECOMPOSE" in text:
        return "DECOMPOSE"
    if "SEMANTIC" in text:
        return "SEMANTIC"

    return "SEMANTIC"  # safe default


# Query routing
def route_query(query: str) -> str:
    q = query.lower()

    route = normalize_route(query_classifier(query, call_llm))

    if route == "KEYWORD":
        return "keyword"

    if route == "DECOMPOSE":
        return "decompose"

    return "semantic"

# Heading detection
def is_heading(line: str) -> bool:
    line = line.strip()
    if len(line) < 5 or len(line) > 80:
        return False
    if line.isupper():
        return True
    if line.istitle():
        return True
    return False

# Document topic extraction
def extract_document_topics(call_llm, full_text: str, max_topics: int = 8) -> List[str]:
    prompt = f"""
You are analyzing a document.

Your task:
Identify up to {max_topics} high-level topics that this document discusses.

Rules:
- Topics must be short phrases (1–3 words)
- Topics must come ONLY from the document content
- Do NOT invent topics
- Do NOT explain anything
- One topic per line

Document excerpt:
{full_text[:3000]}
"""
    response = call_llm(prompt)
    return [t.strip().lower() for t in response.split("\n") if t.strip()]

# Chunk topic inference
def infer_chunk_topic(call_llm, chunk_text: str, doc_topics: List[str]) -> str:
    topic_list = "\n".join(doc_topics)

    prompt = f"""
Assign ONE topic from the list below that best matches the text.

Rules:
- Choose ONLY from the provided topics
- Return EXACT topic text
- No explanations

Topics:
{topic_list}

Text:
{chunk_text[:800]}
"""
    return call_llm(prompt).strip().lower()

# PDF extraction with page awareness
def extract_pages_with_sections(path: str):
    reader = PdfReader(path)
    pages = []
    current_section = "unknown"

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue

        lines = text.split("\n")
        for line in lines:
            if is_heading(line):
                current_section = line.strip()
                break

        pages.append({
            "page": i + 1,
            "section": current_section,
            "text": text
        })

    return pages



def chunk_page_with_metadata(page_text, page, section, call_llm, doc_topics):
    chunks = []
    start = 0
    chunk_id = 0

    while start < len(page_text):
        end = start + 1000
        chunk_text = page_text[start:end]

        topic = infer_chunk_topic(call_llm, chunk_text, doc_topics)

        chunks.append({
            "text": chunk_text,
            "metadata": {
                "page": page,
                "section": section,
                "topic": topic,
                "chunk_id": chunk_id
            }
        })

        start = end - 200
        chunk_id += 1

    return chunks



def build_chunks(path: str, call_llm) -> List[Dict]:
    pages = extract_pages_with_sections(path)

    full_text = "\n\n".join(p["text"] for p in pages)
    print("Extracting document topics...")
    doc_topics = extract_document_topics(call_llm, full_text)

    all_chunks = []

    for p in pages:
        page_chunks = chunk_page_with_metadata(
            page_text=p["text"],
            page=p["page"],
            section=p["section"],
            call_llm=call_llm,
            doc_topics=doc_topics
        )
        all_chunks.extend(page_chunks)

    return all_chunks


# Cross-encoder reranker
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
reranker = CrossEncoder(RERANK_MODEL_NAME, device="cpu")


def rerank(query: str, docs: List[str], top_n: int = 5) -> List[str]:
    pairs = [(query, doc) for doc in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(scores, docs),
        key=lambda x: x[0],
        reverse=True
    )

    return [doc for _, doc in ranked[:top_n]]


# Sparse retrieval (BM25)
def tokenize(text: str) -> List[str]:
    return text.lower().split()


class BM25Retriever:
    def __init__(self, documents: List[str]):
        self.documents = documents
        tokenized_docs = [tokenize(d) for d in documents]
        self.bm25 = BM25Okapi(tokenized_docs)

    def retrieve(self, query: str, k: int = 10) -> List[str]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(
            zip(scores, self.documents),
            key=lambda x: x[0],
            reverse=True
        )
        return [doc for _, doc in ranked[:k]]

# Topic inference for queries
def infer_query_topic(call_llm, query: str, doc_topics: List[str]) -> str:
    topic_list = "\n".join(doc_topics)

    prompt = f"""
Choose the ONE topic from the list that best matches the query.

Rules:
- Use ONLY the provided topics
- Return EXACT topic text
- No explanation

Topics:
{topic_list}

Query:
{query}
"""
    return call_llm(prompt).strip().lower()


# Chroma vector store
CHROMA_PATH = "./chroma_db"

client = chromadb.Client(
    Settings(
        persist_directory=CHROMA_PATH,
        anonymized_telemetry=False
    )
)

collection = client.get_or_create_collection(name="microsoft_report")


def add_chunks_to_chroma(chunks: List[Dict]) -> None:
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [str(i) for i in range(len(chunks))]

    embeddings = embed_texts(texts)

    collection.add(
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
        ids=ids
    )


def dense_retrieve(
    query: str,
    call_llm,
    doc_topics: List[str],
    k: int = 10
) -> List[Dict]:

    query_embedding = embed_query(query)
    topic = infer_query_topic(call_llm, query, doc_topics)

    where = None
    if topic and topic in doc_topics:
        where = {"topic": topic}

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=k,
        where=where
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    return [
        {"text": d, "metadata": m}
        for d, m in zip(docs, metas)
    ]

# Query decomposition
def decompose(call_llm, query: str, n: int = 3) -> List[str]:
    prompt = f"""
Split the following question into at most {n} independent sub-questions.
Return each sub-question on a new line.
If it cannot be split, return the original question.

Question:
{query}
"""
    text = call_llm(prompt)
    return [q.strip() for q in text.split("\n") if q.strip()]


# Hybrid retrieval
def hybrid_retrieve(
    query: str,
    bm25: BM25Retriever,
    call_llm,
    doc_topics: List[str],
    k_dense: int = 10,
    k_sparse: int = 10
) -> List[Dict]:

    dense_docs = dense_retrieve(query, call_llm, doc_topics, k_dense)
    sparse_docs = bm25.retrieve(query, k_sparse)

    seen = set()
    combined = []

    for item in dense_docs:
        text = item["text"]
        if text not in seen:
            combined.append(item)
            seen.add(text)

    for text in sparse_docs:
        if text not in seen:
            combined.append({"text": text, "metadata": {}})
            seen.add(text)

    return combined



def query_routing_retrieve(
    query: str,
    bm25: BM25Retriever,
    call_llm,
    doc_topics: List[str]
) -> List[Dict]:

    route = route_query(query)

    if route == "keyword":
        return [{"text": d, "metadata": {}} for d in bm25.retrieve(query, 10)]

    if route == "decompose":
        sub_questions = decompose(call_llm, query)
        all_docs = []

        for sub_q in sub_questions:
            all_docs.extend(
                hybrid_retrieve(sub_q, bm25, call_llm, doc_topics)
            )

        seen = set()
        unique = []

        for d in all_docs:
            if d["text"] not in seen:
                unique.append(d)
                seen.add(d["text"])

        return unique

    return hybrid_retrieve(query, bm25, call_llm, doc_topics)



def main():

    print("Extracting document data...")
    pages = extract_pages_with_sections("data/microsoft-annual-report.pdf")
    full_text = "\n\n".join(p["text"] for p in pages)
    doc_topics = extract_document_topics(call_llm, full_text)

    print("Loading and chunking document...")
    chunks = build_chunks("data/microsoft-annual-report.pdf", call_llm)
    print(f"Total chunks: {len(chunks)}")

    print("Indexing into vector store...")
    add_chunks_to_chroma(chunks)

    bm25 = BM25Retriever([c["text"] for c in chunks])

    query = "Compare Microsoft’s AI strategy with its cloud strategy."
    print(f"\nQuery: {query}")

    candidates = query_routing_retrieve(query, bm25, call_llm, doc_topics)
    print(f"Retrieved {len(candidates)} candidate chunks")

    texts_for_rerank = [c["text"] for c in candidates]
    top_texts = rerank(query, texts_for_rerank)

    context = "\n\n".join(top_texts)

    prompt = f"""
Answer ONLY using the context below.
If the answer is not present, say "Not found".

Context:
{context}

Question:
{query}
"""

    answer = call_llm(prompt)
    print("\nAnswer:\n", answer)


if __name__ == "__main__":
    main()
