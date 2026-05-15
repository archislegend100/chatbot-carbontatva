#!/bin/bash
# scripts/setup_system.sh
# ========================
# One-time system setup: Tesseract OCR + Ollama + Ollama model.
# Run this with sudo/root access.
#
# Usage:
#   chmod +x scripts/setup_system.sh
#   sudo bash scripts/setup_system.sh
#   sudo bash scripts/setup_system.sh --model llama3.2:3b   # smaller model

set -e
OLLAMA_MODEL="${1:-qwen2.5:7b}"
# Parse --model flag
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) OLLAMA_MODEL="$2"; shift ;;
    esac
    shift
done

echo ""
echo "========================================================"
echo " Industrial Energy Efficiency Copilot — System Setup"
echo "========================================================"
echo ""

# ---- 1. Tesseract OCR ----
echo "▶ Installing Tesseract OCR..."
apt-get update -qq
apt-get install -y tesseract-ocr tesseract-ocr-eng libtesseract-dev 2>&1
echo "✅ Tesseract: $(tesseract --version 2>&1 | head -1)"

# ---- 2. Ollama ----
echo ""
echo "▶ Installing Ollama..."
if command -v ollama &>/dev/null; then
    echo "  Ollama already installed: $(ollama --version 2>&1 | head -1)"
else
    curl -fsSL https://ollama.com/install.sh | sh
    echo "✅ Ollama installed"
fi

# ---- 3. Start Ollama service ----
echo ""
echo "▶ Starting Ollama service..."
systemctl enable ollama 2>/dev/null || true
systemctl start ollama 2>/dev/null || (ollama serve &>/dev/null & sleep 3)
echo "✅ Ollama service running"

# ---- 4. Pull model ----
echo ""
echo "▶ Pulling model: $OLLAMA_MODEL"
echo "  (This may take a while depending on your internet speed)"
ollama pull "$OLLAMA_MODEL"
echo "✅ Model $OLLAMA_MODEL ready"

# ---- 5. Verify ----
echo ""
echo "▶ Verification:"
echo "  Tesseract: $(tesseract --version 2>&1 | head -1)"
echo "  Ollama:    $(ollama --version 2>&1 | head -1)"
echo "  Models:    $(ollama list 2>/dev/null | tail -n +2 | head -5)"
echo ""
echo "========================================================"
echo "✅ System setup complete!"
echo ""
echo "Next steps:"
echo "  1. Install Python deps: .venv/bin/pip install -r backend/requirements.txt"
echo "  2. Run ingestion:       ./run_ingest.sh --test-run"
echo "  3. Start backend:       ./run_backend.sh"
echo "  4. Start frontend:      cd frontend && npm run dev"
echo "========================================================"
