# Codebase Structure

This document outlines the current architecture and internal directory structure of the CarbonTatva Industrial Energy Efficiency Copilot. The repository is organized into a completely decoupled Frontend (Next.js) and Backend (FastAPI).

---

## 1. Backend (`/backend/app`)

The backend follows a modular, modular design prioritizing low-latency routing and memory-efficient execution (specifically designed for constrained cloud environments like Render Free Tier).

### Core Components

- **`main.py`**: The primary FastAPI entrypoint. Contains the API routing (`/chat`, `/chat/title`, `/health`), CORS configuration, and the overarching execution pipeline that stitches together retrieval, fusion, and generation.
- **`config/settings.py`**: Centralized configuration management using `pydantic-settings`. Reads `.env` variables and provides typed settings to the rest of the app.

### Generation (`/generation`)
Responsible for interacting with the Large Language Model.
- **`mistral_client.py`**: An asynchronous, lightweight HTTP wrapper around the Mistral API for text generation and batch embedding. Replaced the memory-heavy local `transformers` pipeline.
- **`prompts.py`**: Contains the `SYSTEM_PROMPT` and `USER_PROMPT_TEMPLATE`. Defines the strict engineering tone, rules for handling out-of-domain queries, and formatting.
- **`verifier.py`**: An optional post-generation step that uses the LLM to verify if its own generated answer is fully supported by the retrieved context.

### Retrieval (`/retrieval`)
Handles all semantic, keyword, and hybrid search mechanisms.
- **`dense_retriever.py`**: Performs semantic vector search.
- **`sparse_retriever.py`**: Performs keyword-based BM25 search.
- **`colbert_retriever.py`**: Late-interaction retrieval for ultra-high precision (lazy-loaded to conserve memory).
- **`hybrid_retriever.py`**: Implements Reciprocal Rank Fusion (RRF) to merge and normalize scores from the dense and sparse retrievers.
- **`reranker.py`**: A Cross-Encoder that re-scores the fused candidates for final relevancy (lazy-loaded).
- **`context_compressor.py`**: Extracts only the most relevant sentences from large chunks to reduce token usage before sending context to the LLM.
- **`multi_query.py`**: Expands a single user query into multiple variants to increase search recall.

### Routing & Planning (`/routing`)
Determines *how* a query should be processed before any heavy computation occurs.
- **`query_router.py`**: Analyzes the query to detect intent (e.g., calculation, troubleshooting, fact-finding) and domain (Thermal vs. Electrical).
- **`retrieval_planner.py`**: Consumes the router's profile and the user's `retrieval_mode` (Auto, Fast, Deep, Research) to generate an execution plan (e.g., turning off reranking for "Fast" queries).

### Indexing & Models (`/indexing` & `/models`)
- **`vector_store.py`**: Abstraction layer for interacting with ChromaDB. Specifically designed to prevent automatic background model downloads on server startup.
- **`chat.py`**: Pydantic models representing the exact JSON schemas for frontend-backend communication (e.g., `ChatRequest`, `QueryResponse`, `TitleRequest`).
- **`schemas.py`**: Internal data structures representing chunks, metadata, and document hierarchies.

---

## 2. Frontend (`/frontend`)

The frontend is a modern Next.js (App Router) application utilizing React, Tailwind CSS, and Lucide icons for a premium, ChatGPT-style user interface.

### Core Components

- **`src/app/page.tsx`**: The monolithic core of the application. It manages:
  - **State**: User inputs, chat messages, active UI modes, and expanded agentic thought logs.
  - **Session Management**: Automatically saves and loads chat history via browser `localStorage`.
  - **Intelligent Naming**: Interacts with the backend `/chat/title` API to summarize new conversations.
  - **Layout Constraints**: Handles scrolling mechanics, dark mode toggling, and the responsive sidebar.
- **`src/app/layout.tsx`**: The root HTML layout, injecting global styles and fonts.
- **`src/lib/api.ts`**: Helper utilities for typing and standardizing `fetch` requests sent to the backend.

---

## 3. Deployment Constraints & Solutions

The codebase was heavily refactored from a local-first design to a cloud-native design due to 512MB RAM limits on standard free tiers (like Render).
1. **Lazy Loading**: Libraries like `sentence-transformers` and `torch` are dynamically imported only when specifically required by deep-search pipelines, preventing Out-Of-Memory (OOM) crashes on startup.
2. **CORS Handling**: Cross-Origin Resource Sharing is strictly defined via the `FRONTEND_ORIGIN` environment variable, ensuring the backend safely accepts POST requests from the Vercel-hosted frontend.
