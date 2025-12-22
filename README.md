# Retrieval-Augmented Generation (RAG) From Scratch

## Introduction

This repository contains a **from-scratch implementation of a Retrieval-Augmented Generation (RAG) pipeline**, built to understand how retrieval actually works under the hood — not to hide it behind frameworks.

The goal of this project is **learning and evaluation**, not production readiness.

Large Language Models are good at generating text, but bad at knowing your documents.  
In practice, most RAG failures come from **retrieval**, not the LLM.

This project focuses on understanding and fixing those failures.

---

## What Is RAG (Briefly)

Retrieval-Augmented Generation improves LLM outputs by:

- Retrieving relevant pieces of external data (documents)
- Feeding only those pieces to the LLM
- Generating grounded answers instead of hallucinations

---

## What This Project Does

This pipeline:

- Loads a real PDF document (Microsoft Annual Report)
- Chunks the document
- Indexes chunks in a vector database
- Uses **hybrid retrieval**:
  - Dense retrieval (embeddings)
  - Sparse retrieval (BM25)
- Merges and reranks results using a **cross-encoder**
- Sends grounded context to an LLM via **OpenRouter**
- Returns an answer **only if supported by retrieved context**
- Responds with `"Not found"` if the answer is not present

---

## Tech Stack

### Core
- Python
- NumPy

### Retrieval
- **Dense embeddings:** `BAAI/bge-small-en-v1.5` (local, CPU)
- **Sparse retrieval:** BM25 (`rank-bm25`)
- **Hybrid retrieval:** Dense + Sparse merge
- **Reranking:** Cross-encoder (`ms-marco-MiniLM-L-6-v2`)

### Vector Database
- ChromaDB (local persistence)

### LLM
- OpenRouter API  
- Model: `mistralai/mistral-7b-instruct`

### Utilities
- sentence-transformers
- pypdf
- python-dotenv

---

## Current Capabilities

- End-to-end RAG pipeline
- Hybrid retrieval (dense + sparse)
- Cross-encoder reranking
- Grounded answer generation
- Hallucination-safe prompting
- Local embeddings (no paid embedding APIs)

---

## Known Limitations (Work in Progress)

These are **intentional gaps**, not unknown issues:

- No query decomposition for multi-intent questions
- No query routing based on question type
- No metadata-aware chunking or retrieval
- Simple character-based chunking
- Re-ingestion on every run (no deduplication yet)

These will be addressed incrementally in future iterations.

## Setup Instructions

### Clone the Repository

Clone the repository and move into the project directory:

```bash
git clone <repo-url>
cd <repo-name>
```

### Create a Virtual Environment

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate    # Linux / Mac
venv\Scripts\activate       # Windows
```

### Install Dependencies

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Set Up Environment Variables

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_api_key_here
```

### Add the PDF

Place the Microsoft Annual Report PDF at the following path:

```text
data/microsoft-annual-report.pdf
```

### Run the Pipeline

Run the RAG pipeline:

```bash
python advanced_rag.py
```

## Example Query

```text
What role does natural language play in Microsoft’s AI vision?
```

The system will:

- Retrieve relevant chunks
- Rerank them
- Generate a grounded answer from context

---

## Why This Project Exists

Most RAG examples:

- Hide retrieval behind abstractions
- Don’t evaluate failures
- Hallucinate confidently

This repo exists to:

- Understand retrieval deeply
- Expose failure modes
- Build intuition before moving to agents or fine-tuning

---

## Next Steps

Planned improvements:

- Query routing
- Query decomposition
- Metadata-aware chunking
- Retrieval debugging tools
- Smarter persistence & deduplication

---

## Disclaimer

This project is for learning and experimentation.  
It is not production-ready and does not claim to be.
