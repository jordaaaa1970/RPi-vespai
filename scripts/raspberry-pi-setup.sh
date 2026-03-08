#!/usr/bin/env bash
# VespAI Raspberry Pi Setup Script
# Handles PEP 668 virtual environment requirements and current VespAI launchers

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

INSTALL_SERVICE=0
if [[ "${1:-}" == "--install-service" ]]; then
    INSTALL_SERVICE=1
fi

echo "🍓 VespAI Raspberry Pi Setup"
echo "=============================="

# Check if we're running from the correct location (home directory recommended)
if [[ "$PWD" == "/vespai" ]] || [[ "$PWD" =~ ^/[^/]*$ ]]; then
    echo "⚠️  Warning: Running from system directory ($PWD)"
    echo "   For best results, clone to your home directory:"
    echo "   cd ~ && git clone https://github.com/jakobzeise/vespai.git"
    echo "   This avoids Git permission issues."
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "💡 Recommended: cd ~ && git clone https://github.com/jakobzeise/vespai.git"
        exit 1
    fi
fi

# Check if we're on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/device-tree/model 2>/dev/null; then
    echo "⚠️  Warning: This doesn't appear to be a Raspberry Pi"
    echo "   Use scripts/setup.py for other systems"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check Python version
python_version=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
echo "📋 Python version: $python_version"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo "❌ Python 3.10+ required. Please upgrade Python."
    exit 1
fi

# Update system packages
echo "📦 Updating system packages..."
sudo apt update

# Install system dependencies
echo "🔧 Installing system dependencies..."
sudo apt install -y python3-full python3-pip python3-venv python3-opencv git git-lfs curl

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "🔨 Creating virtual environment..."
    python3 -m venv .venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Initialize Git LFS and download model files
echo "📥 Setting up Git LFS for model files..."
git lfs install
git lfs pull

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source .venv/bin/activate

# Verify we're in virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

echo "✅ Virtual environment active: $VIRTUAL_ENV"

echo "⬆️  Updating packaging tools..."
python -m pip install --upgrade pip setuptools wheel

# Run main setup script
echo "🚀 Running VespAI setup..."
python scripts/setup.py

echo "🔧 Updating launcher permissions..."
chmod +x start_vespai.sh start_vespai_web.sh scripts/quick-start.sh

if [[ $INSTALL_SERVICE -eq 1 ]]; then
    echo "⚙️  Installing boot-time web service..."
    sudo cp "$PROJECT_DIR/vespai-web.service" /etc/systemd/system/vespai-web.service
    sudo systemctl daemon-reload
    sudo systemctl enable vespai-web.service
    sudo systemctl restart vespai-web.service
    echo "✅ Boot service installed: vespai-web.service"
fi

echo ""
echo "🎉 Raspberry Pi setup complete!"
echo ""
echo "🔧 To use VespAI:"
echo "   1. Activate virtual environment: source .venv/bin/activate"
echo "   2. Run VespAI: ./start_vespai_web.sh"
echo "   3. Open http://$(hostname -I | awk '{print $1}'):8081 in browser"
echo ""
echo "💡 For best performance on Raspberry Pi:"
echo "   - Use --resolution 720p or 640x480"
echo "   - Enable --motion detection"
echo "   - Set GPU memory to 128MB+ with sudo raspi-config"
echo "   - Configure .env with RESOLUTION=1280x720, FRAME_DELAY=0.35, DATASET_FRAME_DELAY=5.0"
echo ""
echo "🔁 Optional boot service:"
echo "   Run this script with --install-service to enable auto-start after reboot"
echo "   Logs: sudo journalctl -u vespai-web.service -f"
echo ""