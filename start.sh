#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Load .env if present ──────────────────────────────────────────────────────
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

# ── Check API key ─────────────────────────────────────────────────────────────
if [ -z "$SONIOX_API_KEY" ]; then
  echo ""
  echo "  ERROR: SONIOX_API_KEY is not set."
  echo ""
  echo "  Either:"
  echo "    export SONIOX_API_KEY=your_key_here"
  echo "  or create a .env file:"
  echo "    echo 'SONIOX_API_KEY=your_key_here' > .env"
  echo ""
  exit 1
fi

# ── Activate virtual environment ──────────────────────────────────────────────
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt --quiet
fi

source .venv/bin/activate

# ── Build index if not present ────────────────────────────────────────────────
if [ ! -f "quran_index.pkl" ]; then
  echo "Building Quran index (first run only)..."
  python build_index.py
fi

# ── Start server ──────────────────────────────────────────────────────────────
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

echo ""
echo "  Quran Tracker"
echo "  ─────────────────────────────────"
echo "  Server:   http://$HOST:$PORT"
echo "  Debug UI: http://$HOST:$PORT/"
echo "  Health:   http://$HOST:$PORT/health"
echo "  Search:   http://$HOST:$PORT/search?q=بسم الله"
echo "  ─────────────────────────────────"
echo ""

# Open browser after a short delay (macOS)
if command -v open &>/dev/null; then
  (sleep 1.5 && open "http://$HOST:$PORT/") &
fi

uvicorn server:app \
  --host "$HOST" \
  --port "$PORT" \
  --log-level info
