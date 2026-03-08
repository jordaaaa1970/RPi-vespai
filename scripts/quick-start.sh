#!/usr/bin/env bash
# VespAI Quick Start Script for Linux/macOS
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_DIR"

echo "========================================"
echo "VespAI Hornet Detection System"  
echo "========================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "ERROR: Python not found! Please install Python 3 first."
    echo "Linux: sudo apt install python3 python3-pip"
    echo "macOS: brew install python3"
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

echo "[1/3] Running automated setup..."
$PYTHON_CMD scripts/setup.py

echo ""
echo "[2/3] Setup completed successfully!"
echo "[3/3] Starting VespAI web interface..."
echo ""
echo "Open your browser to: http://localhost:8081"
echo "Press Ctrl+C to stop the server"
echo ""

if [ -x "$PROJECT_DIR/start_vespai_web.sh" ]; then
    exec "$PROJECT_DIR/start_vespai_web.sh"
elif [ -x "$PROJECT_DIR/.venv/bin/python" ]; then
    exec "$PROJECT_DIR/.venv/bin/python" vespai.py --web
elif [ -x "$PROJECT_DIR/venv/bin/python" ]; then
    exec "$PROJECT_DIR/venv/bin/python" vespai.py --web
else
    exec "$PYTHON_CMD" vespai.py --web
fi