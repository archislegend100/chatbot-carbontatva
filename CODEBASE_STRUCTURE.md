# CarbonTatva Codebase Structure

This document outlines the minimal, production-optimized structure of the CarbonTatva Industrial Energy Copilot. 

## 1. Project Root Directory

The root directory serves as the project boundary and contains critical configuration files.
*   **`backend/`**: Contains the FastAPI server, inference logic, and vector database indices.
*   **`frontend/`**: Contains the Next.js React application and UI.
*   **`data/`**: (Local only) Stores the persisted ChromaDB sqlite file, the BM25 index, and raw JSONL chunk data.
*   **`evaluation/`**: Contains synthetic datasets and evaluation scripts.
*   **`render.yaml`**: The Infrastructure-as-Code (IaC) configuration for deploying the backend to Render.
*   **`run_ingest.sh`**: A helper script to orchestrate the ingestion of raw PDFs into the vector databases.

---

## 2. Backend (`/backend`)

The backend is built with **FastAPI** and is designed to be completely stateless, flat, and extremely fast to comply with 512MB free-tier hosting limits.

### Core Entrypoints
*   **`app/main.py`**: The monolithic entrypoint. It initializes the FastAPI server, configures CORS for Vercel communication, and contains the singular `/chat` and `/chat/title` endpoints. It orchestrates the entire RAG pipeline from query to generation.
*   **`app/config/settings.py`**: The global configuration object (`pydantic_settings`). It handles all feature toggles (`ENABLE_COLBERT`, `ENABLE_MULTI_QUERY`) and API keys.

### Retrieval Module (`app/retrieval/`)
This folder contains the specific retrieval logic used to fetch data from the databases.
*   **`dense_retriever.py`**: Interacts with ChromaDB to fetch semantically similar chunks.
*   **`sparse_retriever.py`**: Interacts with the BM25 index to fetch keyword-matching chunks.
*   **`hybrid_retriever.py`**: Merges results from the dense and sparse retrievers using Reciprocal Rank Fusion (RRF).
*   **`multi_query.py`**: Logic for generating alternative queries (HyDE and Multi-Query) to expand the search space.
*   **`parent_expansion.py`**: Automatically fetches the surrounding paragraphs for a highly-ranked chunk to provide better context to the LLM.
*   **`reranker.py`**: A cross-encoder used to re-score the fetched chunks (currently disabled for memory constraints).

### Routing & Planning Module (`app/routing/`)
*   **`query_router.py`**: Analyzes the incoming user query and determines its intent (e.g., "formula", "troubleshoot", "comparison") and domain.
*   **`retrieval_planner.py`**: Generates a dynamic execution plan based on the user's selected mode (`fast`, `deep`, `research`) and the detected query intent. It determines how many chunks to fetch and whether to use advanced features like Multi-Query.

### Generation Module (`app/generation/`)
*   **`mistral_client.py`**: The dedicated client for interacting with the Mistral API (`mistral-small-latest`).
*   **`prompts.py`**: The "Source of Truth" for the AI's persona. It contains the strict `SYSTEM_PROMPT` that enforces accurate citation, mathematical LaTeX formatting, and genuine engineering responses.
*   **`verifier.py`**: A secondary verification step (optional) to ensure the generated answer doesn't hallucinate facts outside the retrieved context.

### Indexing & Ingestion Modules (`app/indexing/`, `app/ingestion/`)
These modules are strictly used by `scripts/ingest.py` when running `./run_ingest.sh` to update the database.
*   **`pdf_loader.py` & `pdf_ocr_loader.py`**: Extracts text from the BEE manuals.
*   **`chunker.py`**: Slices the text into overlapping semantic chunks.
*   **`vector_store.py` & `bm25_index.py`**: Embeds the chunks and saves them to the `data/` directory.

---

## 3. Frontend (`/frontend`)

The frontend is a **Next.js 16 (App Router)** application built with **React**, **Tailwind CSS**, and **Lucide React**.

*   **`src/app/page.tsx`**: The core UI component. It handles:
    *   **State Management & Isolation**: Manages chat sessions, messages, and UI toggles using React `useState` and `useRef`. Implements isolated session tracking to allow simultaneous query generation across multiple chat sessions without state leakage.
    *   **Local Storage Sync**: Persists chat history across sessions using `localStorage`.
    *   **Dynamic UI**: Renders the "Retrieval Execution Pipeline" debugging block and mathematical formulas (via `react-markdown` + `rehype-katex`).
    *   **Suggested Questions**: Dynamically shuffles and renders clickable suggested questions on empty states.
*   **`src/app/globals.css`**: Tailwind directives and custom scrollbar/markdown styling.
*   **`src/lib/api.ts`**: The Axios-based API client that connects the frontend to the Render backend, handling stream connections and URL sanitization.

---

## 4. Architecture Evolution (V1 vs. V2)

CarbonTatva underwent a significant architectural evolution to reach its current production state.

### V1: The LangGraph Local Prototype
Initially, the system was built as a complex, stateful graph using **LangGraph** and ran locally using **Ollama**. 
*   **Structure:** It utilized deep folder hierarchies (`app/graph`, `app/router`, `app/api`, `app/tools`, `app/core`) to manage state transitions between various agentic nodes.
*   **Problems:** While highly modular, it was extremely slow, bloated, and impossible to host on free-tier cloud providers due to massive memory footprints and state management overhead. 

### V2: The Minimalist Mistral Cloud Pipeline (Current)
To prepare for production deployment on Render and Vercel, the architecture was aggressively stripped down and flattened.
*   **Migration:** The entire LangGraph state machine was discarded in favor of a fast, linear pipeline inside `main.py` orchestrated by a simple `retrieval_planner.py`.
*   **Cloud-Native:** We migrated from local Ollama to the cloud-based Mistral API, achieving sub-second generation times without relying on local hardware.
*   **Cleanup:** Thousands of lines of dead code (old endpoints, graph nodes, complex routers, and local bash scripts) were deleted to establish a "Proper Minimal Repository."
*   **Result:** The system now successfully runs a premium, ChatGPT-like interface via Vercel while executing advanced, dynamic RAG strategies on a 512MB RAM Render instance.
