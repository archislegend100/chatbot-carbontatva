#!/bin/bash
# Run the full PDF ingestion and indexing pipeline.
# Usage: ./run_ingest.sh [--test-run] [--force-rebuild]

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$PROJECT_ROOT/.venv"
BACKEND="$PROJECT_ROOT/backend"
BACKEND_URL="${BACKEND_URL:-http://localhost:8000}"

if [ ! -d "$VENV" ]; then
    echo "ERROR: Virtual environment not found. Run setup first."
    exit 1
fi

echo "Starting PDF ingestion pipeline..."
echo "Processing both BEE PDF manuals. Estimated time: 20-40 min."
echo ""

cd "$PROJECT_ROOT"
PYTHONPATH="$BACKEND" "$VENV/bin/python" scripts/ingest.py "$@"

echo ""
echo "Ingestion complete."

echo "Notifying backend to reload indexes..."
if curl -sf -X POST "$BACKEND_URL/api/reload" -o /dev/null 2>/dev/null; then
    echo "Backend reloaded — index live at $BACKEND_URL"
else
    echo "Backend not running at $BACKEND_URL"
    echo "Start it with: ./run_backend.sh"
fi
