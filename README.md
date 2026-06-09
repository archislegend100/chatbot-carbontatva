# CarbonTatva: Industrial Energy Efficiency Copilot

A minimal, highly optimized RAG (Retrieval-Augmented Generation) copilot designed to assist engineers with industrial energy efficiency, specifically built around the Bureau of Energy Efficiency (BEE) manuals.

## Architecture

CarbonTatva operates on a stateless, cloud-native architecture optimized for speed and free-tier hosting limits:
- **Frontend:** Next.js (React) + Tailwind CSS + Lucide Icons. Hosted on Vercel.
- **Backend:** FastAPI (Python) running a flat, highly-optimized routing pipeline. Hosted on Render.
- **Inference:** Powered by Mistral API (`mistral-small-latest`).
- **Vector Database:** Local ChromaDB (SQLite) + BM25 Sparse Index.

## Quickstart

### 1. Backend Setup (Render)

1. Navigate to the `backend` directory:
   ```bash
   cd backend
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your environment variables (in `.env` or in your Render dashboard):
   - `MISTRAL_API_KEY`: Your Mistral AI API key.
   - `FRONTEND_ORIGIN`: Your deployed Vercel URL (e.g., `https://my-app.vercel.app`) to resolve CORS.
4. Run the backend:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

### 2. Frontend Setup (Vercel)

1. Navigate to the `frontend` directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Set your environment variables (in `.env.local` or Vercel dashboard):
   - `NEXT_PUBLIC_API_URL`: The URL of your deployed Render backend (e.g., `https://carbontatva-backend.onrender.com`).
4. Run the development server:
   ```bash
   npm run dev
   ```

## Ingestion Pipeline

If you need to update the BEE manuals or rebuild the vector index, the ingestion scripts are preserved:
```bash
./run_ingest.sh
```
This requires a local Python environment and extracts data from the PDFs located in the root directory into the `data/` folder.
