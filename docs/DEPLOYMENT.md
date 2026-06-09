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

Render is used for deploying the FastAPI backend as a persistent Web Service. Render does not provide a CLI for initial setup, so you must use their Web Dashboard.

**Step-by-Step Instructions:**
1. **Log In:** Go to [dashboard.render.com](https://dashboard.render.com/) and log in using your GitHub account.
2. **Create New Web Service:** Click the **"New +"** button in the top right corner and select **"Web Service"**.
3. **Connect Repository:** In the list of your connected GitHub repositories, find and select `chatbot-carbontatva`. (If you don't see it, you may need to click "Configure account" to grant Render access to this specific repository).
4. **Configure the Service:**
   - **Name:** Enter a name like `carbontatva-backend`.
   - **Region:** Select a region close to your users (e.g., Singapore or Frankfurt).
   - **Branch:** `master`
   - **Root Directory:** Type `backend` exactly as written.
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. **Select Instance Type:** Choose the **Free** tier (or a paid tier if you expect heavier traffic).
6. **Add Environment Variables:** Scroll down to the "Environment Variables" section and click "Add Environment Variable". Add the following:
   - `MISTRAL_API_KEY`: Paste your Mistral API Key here.
   - `FRONTEND_ORIGIN`: Add your Vercel frontend URL (e.g., `https://chatbot-carbontatva.vercel.app` — no trailing slash).
   - (The rest of the config values like `DEFAULT_RETRIEVAL_MODE` are handled safely by `render.yaml` or defaults).
7. **Deploy:** Click **"Create Web Service"** at the bottom of the page.
8. **Wait for Build:** Render will clone your code, run `pip install`, and start `uvicorn`. Once the logs say "Application startup complete", copy the public `.onrender.com` URL provided at the top left of the dashboard.

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
