#!/bin/bash
# Start the FastAPI backend. Run from the project root.

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$PROJECT_ROOT/.venv"
BACKEND="$PROJECT_ROOT/backend"
PORT="${PORT:-8000}"

if [ ! -d "$VENV" ]; then
    echo "ERROR: Virtual environment not found at $VENV"
    echo "  Run: python3 -m venv .venv && .venv/bin/pip install -r backend/requirements.txt"
    exit 1
fi

# Kill any stale process on the target port
EXISTING_PID=$(lsof -ti:"$PORT" 2>/dev/null || true)
if [ -n "$EXISTING_PID" ]; then
    echo "Port $PORT in use by PID $EXISTING_PID — clearing..."
    kill "$EXISTING_PID" 2>/dev/null || true
    sleep 1
fi

# Check for placeholder API key
LLM_PROVIDER_VAL=$(grep "^LLM_PROVIDER=" "$PROJECT_ROOT/.env" 2>/dev/null | cut -d= -f2 | tr -d '[:space:]')
if [ "$LLM_PROVIDER_VAL" = "mistral" ] && grep -q "MISTRAL_API_KEY=your-mistral-api-key-here" "$PROJECT_ROOT/.env" 2>/dev/null; then
    echo ""
    echo "WARNING: MISTRAL_API_KEY is still a placeholder in .env"
    echo "  Get a key at https://console.mistral.ai and update MISTRAL_API_KEY in .env"
    echo ""
fi

echo "Starting backend on http://localhost:$PORT"
echo "API docs: http://localhost:$PORT/api/docs"
echo ""

cd "$PROJECT_ROOT"
PYTHONPATH="$BACKEND" "$VENV/bin/uvicorn" \
    app.main:app \
    --host "${HOST:-0.0.0.0}" \
    --port "$PORT" \
    --reload \
    --reload-dir "$BACKEND/app" \
    "$@"
