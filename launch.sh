#!/bin/bash
# ChemJEPA Web Interface Launcher

echo "============================================"
echo "🧪 ChemJEPA Web Interface"
echo "============================================"
echo ""

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run ./setup.sh first"
    exit 1
fi

# Activate venv
echo "Activating virtual environment..."
source .venv/bin/activate

# Check if models exist
if [ ! -f "checkpoints/best_jepa.pt" ]; then
    echo "⚠️  Warning: Phase 1 model not found at checkpoints/best_jepa.pt"
    echo "   The interface will work but won't have Phase 1 predictions"
fi

if [ ! -f "checkpoints/production/best_energy.pt" ]; then
    echo "⚠️  Warning: Phase 2 model not found"
    echo "   The interface will work but won't have energy predictions"
fi

echo ""
echo "============================================"
echo "🚀 Starting Web Server..."
echo "============================================"
echo ""
echo "The interface will open at:"
echo ""
echo "  🌐 http://localhost:7860"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "Loading models (this may take 5-10 seconds)..."
echo "============================================"
echo ""

# Launch the interface
python3 interface/app.py
