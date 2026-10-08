# RAGRoute

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.0%2B-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-2.5-4285F4.svg)](https://ai.google.dev/)
[![pgvector](https://img.shields.io/badge/PostgreSQL-pgvector-336791.svg)](https://github.com/pgvector/pgvector)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A high-precision, domain-specific Retrieval-Augmented Generation (RAG) agent pipeline for wine review and tasting note retrieval. Built on **LangGraph**, **Google Gemini**, and **PostgreSQL with pgvector**.

---

## 🌟 Key Features

- **Dual-Agent Architecture**: Decouples query routing and extraction from retrieval execution and synthesis.
- **Dynamic Retrieval Routing**: Automatically chooses between lexical (BM25 full-text search), semantic (vector cosine search), and hybrid search.
- **Strict Domain Guardrails**: Enforces wine-specific knowledge boundaries, rejecting off-topic queries with standardized responses.
- **Deterministic Query Extraction**: Preserves canonical wine labels for exact search and full query context for semantic descriptors.
- **Local Documentation Site**: Scaffolds full documentation using docs7 standards.

---

## 🧠 Architecture Overview

RAGRoute coordinates two specialized Gemini models in a directed LangGraph pipeline:

```mermaid
flowchart TD
  START(["START"]) --> CLASSIFIER["Classifier Node - call_classifier_model"]
  CLASSIFIER --> SYNTH["Synthesizer Node - call_synthesiser_model"]
  SYNTH --> END(["END"])
```

### 1. Classifier Agent (`gemini-2.5-flash-lite`)
- Classifies incoming queries into `lexical`, `semantic`, `hybrid`, or `off_topic`.
- Returns structured JSON: `{ "label", "confidence", "alternatives", "query" }`.
- Applies deterministic query normalization rules.
- Automatically falls back to `hybrid` when confidence is `< 0.7`.

### 2. Synthesizer Agent (`gemini-2.5-flash`)
- Receives the routing decision from conversation state.
- Executes automatic tool calling on vector and lexical indices (`bm25_search`, `dense_search`, `hybrid_search`).
- Generates grounded natural-language answers strictly based on retrieved tasting notes.
- Emits standardized refusals for off-topic inquiries.

---

## 🔍 Deterministic Query Strategies

| Retrieval Mode | Retreiver Tool | Strategy & Target |
| :--- | :--- | :--- |
| **Lexical** | `bm25_search` | Extract canonical wine label (verbatim quoted phrase or stripped question scaffolding) against PostgreSQL `tsvector`. |
| **Semantic** | `dense_search` | Use complete user query against `pgvector` HNSW index with cosine distance. |
| **Hybrid** | `hybrid_search` | Fuse canonical wine label and semantic descriptors into a merged query phrase. |
| **Off-Topic** | *None* | Reject non-wine queries with standardized response. |

---

## 🚀 Quick Start

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/RAGRoute.git
cd RAGRoute

# Using uv (recommended)
uv sync

# Or using pip
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### 2. Configure Environment

Create a `.env` file at the root of the project:

```env
GEMINI_API_KEY=your_gemini_api_key_here
DB_DSN=postgres://postgres:password@localhost:5432/wine_review_text_vector_search_db
```

### 3. Initialize Database

Run PostgreSQL with the `pgvector` extension and execute the ingestion script:

```bash
python -m src.rag_agent.db.batch_insert
```

### 4. Run LangGraph Dev Server

Start the local LangGraph runtime:

```bash
langgraph dev
```

---

## 🧪 Testing

Run test suites using `uv` or `pytest`:

```bash
# Run unit and integration test suites
uv run pytest tests/unit_tests tests/integration_tests

# Run all tests
uv run pytest tests/
```

---

## 📚 Documentation & Guides

- **[docs/](docs/)**: Complete docs7 documentation site (`docs7 dev docs --port 3333`).
  - [Overview](docs/index.mdx)
  - [Installation & Setup](docs/installation.mdx)
  - [Architecture & Graph](docs/architecture.mdx)
  - [Agents & Policy](docs/agents.mdx)
  - [Retrieval & Vector Search](docs/retrieval.mdx)
  - [Development & Testing](docs/development.mdx)
- **[AGENTS.md](AGENTS.md)**: Deep-dive technical specifications, prompt policies, and JSON contracts for autonomous agents.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
