# Industrial Energy Efficiency Copilot (CarbonTatvaAI)

**Agentic RAG system over BEE Thermal and Electrical Utility Manuals.**  
Built on **LangGraph** orchestration · **Mistral API** (cloud LLM) · **Tesseract OCR** · Hybrid RAG · Multi-provider LLM support.

---

## Table of Contents

1. [What It Does](#1-what-it-does)
2. [System Requirements](#2-system-requirements)
3. [One-Time Setup](#3-one-time-setup)
4. [Running the System](#4-running-the-system)
5. [Testing the Pipeline](#5-testing-the-pipeline)
6. [API Reference](#6-api-reference)
7. [Configuration](#7-configuration)
8. [Project Structure](#8-project-structure)
9. [Evaluation Framework](#9-evaluation-framework)
10. [Extending the System](#10-extending-the-system)
11. [Architecture Summary](#11-architecture-summary)
12. [Technical Report](#12-technical-report)

---

## 1. What It Does

The copilot answers questions **exclusively from two BEE manuals**:
- **Energy Efficiency in Thermal Utilities** — boilers, furnaces, steam, combustion, heat recovery, cogeneration
- **Energy Efficiency in Electrical Utilities** — motors, VFDs, power factor, tariffs, lighting, compressed air

Eight specialist tools respond to different query types:

| Tool | Query Type | Example |
|------|-----------|---------| 
| `qa` | Direct factual question | "What is optimum excess air for a coal boiler?" |
| `explainer` | Conceptual explanation | "Explain how a VFD reduces motor energy" |
| `troubleshoot` | Fault diagnosis | "Why is my boiler efficiency below 75%?" |
| `opportunity` | Energy savings | "What are savings in a steam distribution system?" |
| `comparison` | Side-by-side | "Compare fire tube vs water tube boilers" |
| `checklist` | Audit checklists | "Generate a boiler house audit checklist" |
| `navigation` | Find content | "Which chapter covers waste heat recovery?" |
| `summarize` | Summaries | "Summarize energy conservation in furnaces" |

---

## 2. System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| OS | Ubuntu 20.04+ | Ubuntu 22.04 |
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16 GB |
| Disk | 10 GB free | 20 GB |
| Python | 3.11 | 3.12 |
| Node.js | 18 | 20 |
| Mistral API key | Required | — |

> **No GPU required** when using the Mistral API (default). GPU is only needed if you switch to a local Ollama model.

---

## 3. One-Time Setup

### Step 1 — System dependencies (requires sudo)

```bash
# Tesseract OCR (REQUIRED — both PDFs are fully image-scanned)
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng
```

Or use the provided setup script (also installs Ollama if you want a local fallback):
```bash
sudo bash scripts/setup_system.sh
```

### Step 2 — Python virtual environment

```bash
cd /path/to/chatbot_carbontatva
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

### Step 3 — Environment configuration

```bash
cp .env.example .env
# Open .env and set your Mistral API key
```

Minimum required configuration:
```bash
LLM_PROVIDER=mistral
MISTRAL_API_KEY=your-mistral-api-key-here   # Get at https://console.mistral.ai
MISTRAL_MODEL=mistral-small-latest
```

### Step 4 — Frontend dependencies

```bash
cd frontend
npm install
cd ..
```

### Step 5 — Ingest the PDFs (one-time, OCR takes 60–90 min total)

```bash
# Quick test first: OCR first 20 pages per PDF (~5–10 min)
./run_ingest.sh --test-run

# Full ingestion after test passes (run overnight if needed)
./run_ingest.sh
```

> **OCR is cached.** After the first run, all subsequent ingestion calls reuse cached text — rebuilding the index from scratch takes only a few minutes.

---

## 4. Running the System

Open **2 terminals**:

**Terminal 1 — Backend**
```bash
./run_backend.sh
# Backend starts at http://localhost:8000
# Swagger UI: http://localhost:8000/docs
```

**Terminal 2 — Frontend**
```bash
cd frontend && npm run dev
# UI starts at http://localhost:3000
```

**Health check:**
```bash
curl http://localhost:8000/api/health
```
Expected: `{"status":"ok","index_loaded":true,"chunk_count":...}`

> If using Ollama as a local fallback, also start it in a third terminal: `ollama serve`

---

## 5. Testing the Pipeline

### Quick API smoke test

```bash
# 1. Health
curl http://localhost:8000/api/health | python3 -m json.tool

# 2. Single query (non-streaming)
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the optimum excess air for a coal-fired boiler?",
    "tool_mode": "auto",
    "domain_filter": null,
    "explanation_level": "engineer"
  }' | python3 -m json.tool

# 3. Streaming query (SSE)
curl -X POST http://localhost:8000/api/stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "query": "Why might boiler efficiency drop below 75%?",
    "tool_mode": "auto",
    "explanation_level": "engineer"
  }'

# 4. Classify only (fast, no LLM generation)
curl -X POST http://localhost:8000/api/classify \
  -H "Content-Type: application/json" \
  -d '{"query": "Compare fire tube and water tube boilers"}'
```

### OCR quality check

```bash
# Preview OCR on first 10 pages of thermal PDF
python scripts/ocr_extract.py --thermal --pages 0-9 --quality-report

# Save OCR text to file for inspection
python scripts/ocr_extract.py --thermal --pages 0-29 \
  --output data/thermal_sample.txt
```

### Backend unit tests

```bash
source .venv/bin/activate
PYTHONPATH=backend python -m pytest backend/tests/test_graph.py -v   # 18 tests
PYTHONPATH=backend python -m pytest backend/tests/test_ocr.py -v     # 9 tests
PYTHONPATH=backend python -m pytest backend/tests/ -v                 # all
```

### Full automated evaluation

```bash
# Backend must be running first (./run_backend.sh)

# Quick smoke (QA category only, ~5 min)
python evaluation/evaluate.py --category qa --run-name smoke

# Full evaluation (all 29 test cases, ~20–30 min)
python evaluation/evaluate.py --run-name run_001

# Results:
ls evaluation/results/run_001/
#   results.jsonl   — per-case detail
#   results.csv     — spreadsheet-friendly
#   report.md       — human-readable report
#   aggregate.json  — machine-readable summary
```

### Manual verification in the UI

1. Open http://localhost:3000
2. Check the **KB Ready** indicator (bottom left sidebar)
3. Enable **🔍 Debug** toggle to see planner routing for each query
4. Ask a test query and verify the answer appears in the response panel

**Recommended test queries:**
```
Thermal:    "What is the significance of O₂ percentage in flue gas?"
Electrical: "How does a VFD follow the affinity law to save motor energy?"
Troubleshoot: "Why are our induction motors running hot?"
Checklist:  "Generate an energy audit checklist for a compressed air system"
Comparison: "Compare synchronous and induction motors for large drives"
```

---

## 6. API Reference

### Base URL: `http://localhost:8000`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check, index status, chunk count |
| `GET` | `/api/status` | Detailed system configuration |
| `POST` | `/api/query` | Main query (full response, non-streaming) |
| `POST` | `/api/stream` | Streaming query (SSE token stream) |
| `POST` | `/api/classify` | Classify query, no LLM generation |
| `GET` | `/docs` | Swagger interactive API docs |

### POST `/api/query` — Request

```json
{
  "query": "string",
  "tool_mode": "auto | qa | explainer | troubleshoot | opportunity | comparison | checklist | navigation | summarize",
  "domain_filter": "thermal | electrical | null",
  "explanation_level": "beginner | engineer"
}
```

### POST `/api/query` — Response

```json
{
  "answer": "string",
  "citations": [
    {
      "chunk_id": "string",
      "book_name": "string",
      "chapter_title": "string",
      "section_title": "string",
      "page_start": 42,
      "page_end": 43,
      "relevance_score": 0.91,
      "snippet": "string"
    }
  ],
  "classification": {
    "tool_mode": "qa",
    "utility_domain": "thermal",
    "confidence": 0.88,
    "planner_raw": "{ ... }",
    "planner_error": false
  },
  "node_latency": {
    "planner_router": 1840,
    "retrieve_dense": 95,
    "retrieve_sparse": 22,
    "merge_results": 8,
    "rerank": 340,
    "answer_generation": 5200
  },
  "latency_ms": 7580,
  "follow_up_suggestions": ["string"],
  "tool_mode": "qa",
  "utility_domain": "thermal"
}
```

### POST `/api/stream` — SSE Events

```
data: {"type":"status","content":"Analysing query..."}
data: {"type":"planner","planner":{...},"error":false}
data: {"type":"token","content":"The optimum excess air..."}
data: {"type":"token","content":" for a coal-fired boiler..."}
data: {"type":"done","citations":[...],"latency_ms":7800,...}
data: [DONE]
```

---

## 7. Configuration

All settings in `.env`:

```bash
# === LLM (Primary: Mistral API) ===
LLM_PROVIDER=mistral
MISTRAL_API_KEY=your-mistral-api-key-here   # https://console.mistral.ai
MISTRAL_MODEL=mistral-small-latest

# === Optional cloud alternatives ===
# LLM_PROVIDER=gemini
GEMINI_API_KEY=
GEMINI_MODEL=gemini-1.5-flash

# LLM_PROVIDER=openai
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini

# === Local fallback (no API key needed) ===
# LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b

# === Retrieval ===
DENSE_TOP_K=15          # Candidates from dense search
SPARSE_TOP_K=15         # Candidates from BM25
RERANK_TOP_K=5          # After cross-encoder reranking
HYBRID_ALPHA=0.6        # 1.0 = pure dense, 0.0 = pure BM25

# === OCR ===
OCR_DPI=200             # Render DPI for Tesseract
FORCE_OCR=false         # Re-run OCR even if cache exists

# === Generation ===
MAX_GENERATION_TOKENS=1200
MAX_CONTEXT_CHUNKS=5

# === PDFs (relative to project root) ===
PDF_THERMAL=bee guide - thermal utility.pdf
PDF_ELECTRICAL=bee guide - electrical utilities.pdf
```

### Supported LLM providers

| Provider | Key Setting | Model Setting | Notes |
|----------|------------|---------------|-------|
| `mistral` | `MISTRAL_API_KEY` | `MISTRAL_MODEL` | **Default — fast, cloud API** |
| `gemini` | `GEMINI_API_KEY` | `GEMINI_MODEL` | Google Gemini |
| `openai` | `OPENAI_API_KEY` | `OPENAI_MODEL` | OpenAI GPT |
| `ollama` | _(none)_ | `OLLAMA_MODEL` | Local fallback, requires GPU |

### Ollama model options (local fallback)

| Model | VRAM | Quality | Speed | When to use |
|-------|------|---------|-------|-------------|
| `qwen2.5:7b` | 4.7 GB | ⭐⭐⭐⭐ | ⭐⭐⭐ | **Recommended local** |
| `llama3.2:3b` | 2.0 GB | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Low VRAM |
| `gemma2:9b` | 5.5 GB | ⭐⭐⭐⭐⭐ | ⭐⭐ | Best local quality |

---

## 8. Project Structure

```
chatbot_carbontatva/
├── .env                            # Your config (NOT committed)
├── .env.example                    # Template
├── pyproject.toml                  # pytest config
├── run_backend.sh                  # One-command backend start
├── run_ingest.sh                   # One-command ingestion
├── Dockerfile.backend
├── docker-compose.yml
│
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI app entrypoint
│   │   ├── api/
│   │   │   ├── health.py           # GET /api/health, /api/status
│   │   │   ├── query.py            # POST /api/query
│   │   │   ├── stream.py           # POST /api/stream (SSE)
│   │   │   └── ingest.py           # POST /api/ingest
│   │   ├── core/
│   │   │   ├── config.py           # Pydantic settings
│   │   │   └── logging.py
│   │   ├── graph/                  # LangGraph pipeline
│   │   │   ├── state.py            # CopilotState TypedDict
│   │   │   ├── planner.py          # Structured JSON router
│   │   │   ├── nodes.py            # Graph nodes
│   │   │   └── graph.py            # Compiled graph
│   │   ├── router/
│   │   │   ├── query_classifier.py # Rule-based + LLM classifier
│   │   │   └── tool_router.py      # Tool dispatch logic
│   │   ├── ingestion/
│   │   │   ├── pdf_loader.py       # Native-text PDF loader
│   │   │   ├── pdf_ocr_loader.py   # OCR-based PDF loader
│   │   │   ├── ocr_engine.py       # Tesseract wrapper
│   │   │   ├── ocr_cache.py        # Page-level JSONL cache
│   │   │   ├── structure_parser.py # Chapter/section hierarchy
│   │   │   └── chunker.py          # Multi-granular chunking
│   │   ├── indexing/
│   │   │   ├── embedder.py         # all-MiniLM-L6-v2
│   │   │   ├── vector_store.py     # ChromaDB wrapper
│   │   │   └── bm25_index.py       # BM25 index
│   │   ├── retrieval/
│   │   │   ├── dense_retriever.py
│   │   │   ├── sparse_retriever.py
│   │   │   ├── hybrid_retriever.py # RRF fusion
│   │   │   └── reranker.py         # Cross-encoder
│   │   ├── generation/
│   │   │   ├── llm_client.py       # Multi-provider adapter
│   │   │   ├── mistral_client.py   # Mistral API client (primary)
│   │   │   ├── ollama_client.py    # Ollama client (local fallback)
│   │   │   └── prompts.py          # Tool-specific prompts
│   │   ├── cache/
│   │   │   └── query_cache.py      # Disk LRU cache
│   │   ├── models/
│   │   │   └── schemas.py          # Pydantic schemas
│   │   ├── services/
│   │   │   └── __init__.py         # Singleton service
│   │   └── utils/
│   ├── tests/
│   │   ├── test_graph.py           # LangGraph tests
│   │   ├── test_ocr.py             # OCR tests
│   │   └── test_ingestion.py       # Ingestion tests
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx          # Root layout + metadata
│   │   │   ├── page.tsx            # Streaming chat UI
│   │   │   └── globals.css         # Design system
│   │   ├── types/api.ts
│   │   └── lib/api.ts              # SSE streaming client
│   └── package.json
│
├── scripts/
│   ├── ingest.py                   # OCR + index pipeline
│   ├── ocr_extract.py              # Standalone OCR CLI
│   ├── eval_queries.py             # Evaluation query runner
│   └── setup_system.sh             # Tesseract + Ollama setup
│
├── evaluation/
│   ├── evaluate.py                 # Automated eval script
│   └── synthetic_dataset.json      # 29 test cases
│
├── docs/
│   └── technical_report.tex        # Full LaTeX technical report
│
└── data/
    ├── indexes/                    # Generated indexes (gitignored)
    │   ├── chroma/
    │   ├── bm25_index.pkl
    │   └── chunks.jsonl
    ├── ocr_cache/                  # Tesseract page cache (gitignored)
    └── query_cache/                # Response LRU cache (gitignored)
```

---

## 9. Evaluation Framework

### Run evaluation

```bash
# Backend must be running
./run_backend.sh

# Full evaluation (all 29 test cases)
python evaluation/evaluate.py --run-name run_001

# Single category
python evaluation/evaluate.py --category troubleshoot

# Custom K, custom backend
python evaluation/evaluate.py --k 8 --base-url http://localhost:8000
```

### Metrics computed

| Group | Metrics |
|-------|---------|
| Retrieval | Recall@K, Precision@K, MRR, Keyword Coverage |
| Routing | Tool accuracy, Domain accuracy, Planner confidence |
| Answer | Keyword overlap, Completeness, Structure adherence |
| Citation | Presence, Field completeness, Source hint match |
| Latency | Planner, Retrieval, Rerank, Generation, Total (p50/p90) |

### Performance thresholds

| Score | Status |
|-------|--------|
| ≥ 0.75 | Excellent — production ready |
| 0.60–0.75 | Good — minor tuning needed |
| 0.45–0.60 | Fair — check retrieval/prompts |
| < 0.45 | Poor — re-run ingestion or check config |

### Diagnosis quick-reference

| Symptom | Fix |
|---------|-----|
| Tool accuracy < 0.70 | Check planner prompt; verify LLM API key is set |
| Recall@K < 0.40 | Re-run `./run_ingest.sh`; check OCR quality |
| Citation presence < 0.60 | Index empty — run `./run_ingest.sh --force-rebuild` |
| Keyword overlap < 0.35 | Review system prompts in `generation/prompts.py` |
| Total p90 > 20s | Check network/API latency; enable query cache |

---

## 10. Extending the System

### Add a new tool mode

1. Add value to `ToolMode` enum in `backend/app/models/schemas.py`
2. Add keyword triggers to `graph/planner.py` → `_rule_based_plan()`
3. Add system prompt to `generation/prompts.py`
4. Add UI entry to `TOOL_MODES` in `frontend/src/app/page.tsx`
5. Add test cases to `evaluation/synthetic_dataset.json`

### Switch LLM provider

```bash
# .env — switch to Gemini
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-key

# .env — switch to local Ollama
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5:7b
```

### Add more PDFs

1. Place PDF in project root
2. Add entry to `DOCUMENTS` list in `scripts/ingest.py`
3. Run `./run_ingest.sh --force-rebuild`

### Docker deployment

```bash
docker-compose up --build
```

Frontend: http://localhost:3000  |  Backend: http://localhost:8000

---

## 11. Architecture Summary

```
User Query (natural language)
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│               Frontend (Next.js 14)                     │
│  SSE streaming · Debug panel · KB Ready indicator       │
└────────────────────────┬────────────────────────────────┘
                         │ POST /api/stream (SSE)
                         ▼
┌─────────────────────────────────────────────────────────┐
│              LangGraph Pipeline                          │
│                                                         │
│  normalize_query → planner_router                       │
│                         ↓                               │
│           ┌─────────────┴──────────────┐                │
│           ▼                            ▼                │
│    retrieve_dense               retrieve_sparse          │
│   (ChromaDB HNSW)              (BM25 index)             │
│           └─────────────┬──────────────┘                │
│                         ▼                               │
│              merge_results (RRF fusion)                 │
│                         ↓                               │
│                   rerank (cross-encoder)                │
│                         ↓                               │
│              tool_dispatch → answer_generation          │
│                  (Mistral API / configurable LLM)       │
│                         ↓                               │
│              citation_assembly → response_formatter     │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Knowledge Base (local disk)                 │
│  ChromaDB (dense)  ·  BM25 (sparse)  ·  OCR cache      │
│  Indexed from: BEE Thermal + Electrical Manuals (OCR'd) │
└─────────────────────────────────────────────────────────┘
```

**Data flow (ingestion, one-time):**
1. PDF pages → Tesseract OCR at 200 DPI → page-level JSONL cache
2. Pages → Structure parser (chapter/section hierarchy)
3. Sections → Multi-granular chunker (section, semantic, table)
4. Chunks → `all-MiniLM-L6-v2` embeddings → ChromaDB
5. Chunks → BM25 tokeniser → BM25 index (pickle)

**Data flow (query, real-time):**
1. Query → `normalize_query` node
2. Planner calls LLM with JSON mode → structured routing JSON
3. Dense + sparse retrieval run in **parallel** (LangGraph fan-out)
4. RRF merge → cross-encoder rerank → top-5 chunks
5. Tool-specific system prompt + chunks → LLM streaming (Mistral API)
6. Tokens streamed via SSE to frontend in real time
7. Citations assembled from reranked chunks

---

## Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------| 
| Orchestration | LangGraph | ≥0.2 |
| LLM (primary) | Mistral API (`mistral-small-latest`) | cloud |
| LLM (fallback) | Ollama + configurable model | local |
| OCR | Tesseract | ≥5.0 |
| Embeddings | all-MiniLM-L6-v2 | local |
| Vector store | ChromaDB (HNSW) | 0.5.23 |
| Sparse retrieval | rank-bm25 | 0.2.2 |
| Reranker | ms-marco-MiniLM-L-6-v2 | local |
| Backend | FastAPI + uvicorn | 0.115 |
| Frontend | Next.js 14 App Router | 14.x |
| Cache | diskcache | ≥5.6 |
| Testing | pytest + pytest-asyncio | 8.x |

---

## License

Internal prototype — CarbonTatvaAI.  
BEE manuals © Bureau of Energy Efficiency, Government of India.
