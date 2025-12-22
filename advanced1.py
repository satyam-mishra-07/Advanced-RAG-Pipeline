import os
import json
from typing import List
import numpy as np
import requests

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import chromadb
from chromadb.config import Settings

from helper_utils import extract_text_from_pdf

load_dotenv()


# Local embedding model
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


# Simple character-based chunking for PDFs
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    return chunks


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


# Sparse retrieval using BM25
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


# Vector store setup
CHROMA_PATH = "./chroma_db"

client = chromadb.Client(
    Settings(
        persist_directory=CHROMA_PATH,
        anonymized_telemetry=False
    )
)

collection = client.get_or_create_collection(name="microsoft_report")

def add_chunks_to_chroma(chunks: List[str]) -> None:
    embeddings = embed_texts(chunks)
    ids = [str(i) for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=ids
    )

def dense_retrieve(query: str, k: int = 10) -> List[str]:
    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=k
    )

    return results["documents"][0]


# Hybrid retrieval: dense + sparse
def hybrid_retrieve(
    query: str,
    bm25: BM25Retriever,
    k_dense: int = 10,
    k_sparse: int = 10
) -> List[str]:

    dense_docs = dense_retrieve(query, k_dense)
    sparse_docs = bm25.retrieve(query, k_sparse)

    seen = set()
    combined = []

    for doc in dense_docs + sparse_docs:
        if doc not in seen:
            combined.append(doc)
            seen.add(doc)

    return combined


# OpenRouter LLM call
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


def main():
    print("Loading document...")
    text = extract_text_from_pdf("data/microsoft-annual-report.pdf")

    chunks = chunk_text(text)
    print(f"Ingested document into {len(chunks)} chunks")

    print("Indexing chunks...")
    add_chunks_to_chroma(chunks)

    bm25 = BM25Retriever(chunks)

    query = "What role does natural language play in Microsoft’s AI vision?"
    print(f"\nQuery: {query}")

    candidates = hybrid_retrieve(query, bm25)
    print(f"Retrieved {len(candidates)} candidate chunks")

    top_chunks = rerank(query, candidates)
    print("Reranked top chunks")

    context = "\n\n".join(top_chunks)

    prompt = f"""
Answer ONLY using the context below.
If the answer is not present, say "Not found".

Context:
{context}

Question:
{query}
"""

    answer = call_llm(prompt)
    print("\nAnswer:")
    print(answer)


if __name__ == "__main__":
    main()