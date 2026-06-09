# Deployment

This document outlines the deployment strategy for the CarbonTatva Industrial Energy Efficiency Copilot.

## Architecture

The supported and recommended deployment architecture is:
- **Frontend**: Vercel (Next.js)
- **Backend**: Render (FastAPI Web Service)

## 1. Local Development
For local testing, start the backend and frontend separately:
1. Backend: `cd backend && uvicorn app.main:app --reload --port 8000`
2. Frontend: `cd frontend && npm run dev`
*(Ensure `.env` contains `MISTRAL_API_KEY` for the backend and `NEXT_PUBLIC_API_URL=http://localhost:8000` for the frontend)*

## 2. Backend Deployment on Render
Deploy the backend as a Web Service on Render:
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- **Environment Variables**:
  - `MISTRAL_API_KEY`: Your Mistral API Key (Secret)
  - `FRONTEND_ORIGIN`: Your Vercel frontend URL (e.g., `https://your-app.vercel.app`)
  - Optional toggles: `ENABLE_COLBERT`, `ENABLE_HYDE`, `DEFAULT_RETRIEVAL_MODE`, etc.

### Index Persistence Strategy
Do not run OCR/Indexing on every deploy. Build the ChromaDB and BM25 indexes locally and upload them to a persistent disk attached to your Render service (e.g., mounted at `/var/data`). Configure paths accordingly using environment variables (e.g., `INDEX_DIR=/var/data/indexes`). If the index is small, you can optionally bundle it with your code, but a persistent disk is recommended.

## 3. Frontend Deployment on Vercel
Deploy the `frontend` folder to Vercel:
- **Framework Preset**: Next.js
- **Root Directory**: `frontend`
- **Build Command**: `npm run build`
- **Environment Variables**:
  - `NEXT_PUBLIC_API_URL`: Your Render backend public URL (e.g., `https://carbontatva-backend.onrender.com`)

## 4. CORS Setup
The backend automatically configures CORS using the `FRONTEND_ORIGIN` environment variable. Ensure this exactly matches your Vercel URL (without a trailing slash).

## 5. Final Smoke Test
Once both are deployed:
1. Check the backend health: `GET https://your-backend.onrender.com/health`
2. Open the Vercel app and ask a question (e.g., "What is boiler efficiency?"). Verify that the citations and answer appear without CORS errors.
