#!/bin/bash
export PORT=${PORT:-8000}
exec /usr/local/bin/uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1
