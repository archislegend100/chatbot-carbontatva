# CarbonTatva Copilot

CarbonTatva is an industrial energy-efficiency AI assistant built on a Retrieval-Augmented Generation (RAG) architecture. It is designed to provide professional, engineering-focused answers strictly grounded in the Bureau of Energy Efficiency (BEE) Thermal and Electrical Utility manuals.

## Features

- **Grounded Responses**: Exclusively utilizes the Mistral API to generate technically accurate, context-aware answers.
- **Dynamic Retrieval Routing**: Automatically adjusts retrieval depth (auto, fast, deep, research) based on query complexity.
- **Advanced Context Processing**: Uses Reciprocal Rank Fusion (RRF), Cross-Encoder Reranking, and Context Compression.
- **Persistent Chat Sessions**: Features a ChatGPT-style UI with intelligent auto-generated titles, persistent local storage, and history management.
- **Cloud-Native Deployment**: Optimized for serverless environments (Vercel) and constrained memory tiers (Render Free Tier) via lazy-loading of heavy ML models.

## Technology Stack

- **Frontend**: Next.js (React), Tailwind CSS, TypeScript
- **Backend**: FastAPI, Python 3.12+
- **Retrieval Engine**: ChromaDB, BM25, Sentence-Transformers (Lazy-loaded)
- **Generation Model**: Mistral API

## Local Development

### Backend Setup
1. Create a virtual environment: `python -m venv .venv && source .venv/bin/activate`
2. Install dependencies: `pip install -r backend/requirements.txt`
3. Define your API keys in a `.env` file (see `docs/CONFIGURATION.md`).
4. Start the server:
```bash
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Frontend Setup
1. Install Node.js dependencies:
```bash
cd frontend
npm install
```
2. Start the development server:
```bash
npm run dev
```

## Documentation

For a detailed breakdown of the internal architecture and modules, see [CODEBASE_STRUCTURE.md](CODEBASE_STRUCTURE.md).
