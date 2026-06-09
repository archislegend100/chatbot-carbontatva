# CarbonTatva Industrial Energy Efficiency Copilot

A professional, production-style Retrieval-Augmented Generation (RAG) chatbot over the Bureau of Energy Efficiency (BEE) Thermal and Electrical Utility manuals. 

## Features
- **Mistral API** exclusively for high-quality, grounded engineering answers.
- **Advanced Multi-Granular Chunking** (semantic, parent section, table, formula).
- **Hybrid Retrieval** (Dense `bge-large-en-v1.5` + Sparse BM25 with synonym expansion).
- **Reciprocal Rank Fusion (RRF)** + Cross-Encoder Reranking (`bge-reranker-base`).
- **Latency-Aware Execution Flow** (Fast path for simple facts, Advanced path for hard troubleshooting).
- **Context Compression** (Extractive, preserving tables and formulas).

## Tech Stack
- **Backend:** FastAPI, Python 3.10+, LangChain primitives.
- **Retrieval:** ChromaDB, rank-bm25, sentence-transformers.
- **Frontend:** React / Vite.

## Setup and Configuration
See [CONFIGURATION.md](docs/CONFIGURATION.md) for environment variables.
1. Create a virtual environment: `python -m venv .venv && source .venv/bin/activate`
2. Install dependencies: `pip install -r backend/requirements.txt`
3. Set your `MISTRAL_API_KEY` in your environment or `.env` file.

## Running the Backend
```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

## Running the Frontend
```bash
cd frontend
npm install
npm run dev
```

## Evaluation
Run the smoke evaluation script to test the routing logic and system latency:
```bash
python scripts/evaluate.py --smoke
```
Run unit tests:
```bash
pytest backend/tests/
```

## Limitations
This chatbot answers *strictly* from the indexed BEE manuals. It may refuse to answer queries if the required technical evidence is not present in the retrieved context.

## Deployment
Recommended deployment:
- **Frontend**: Vercel
- **Backend**: Render
- **LLM**: Mistral API
- **Vector/Index Storage**: Render persistent disk or prebuilt index artifact

See [DEPLOYMENT.md](docs/DEPLOYMENT.md) for detailed deployment options and strategies.
