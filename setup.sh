#!/bin/bash
# ChemJEPA Complete Setup Script
# This installs everything you need and runs the evaluation

echo "============================================"
echo "🧪 ChemJEPA Complete Setup"
echo "============================================"
echo ""

# Step 1: Activate virtual environment
echo "Step 1/4: Activating virtual environment..."
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found at .venv"
    echo "Creating new virtual environment..."
    python3 -m venv .venv
fi
source .venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Step 2: Upgrade pip
echo "Step 2/4: Upgrading pip..."
pip3 install --upgrade pip --quiet
echo "✓ Pip upgraded"
echo ""

# Step 3: Install dependencies
echo "Step 3/4: Installing dependencies..."
echo "This will take 3-5 minutes. Installing:"
echo "  • PyTorch (deep learning)"
echo "  • PyTorch Geometric (graph neural networks)"
echo "  • RDKit (chemistry)"
echo "  • Scientific packages (numpy, pandas, matplotlib)"
echo "  • Utilities (tqdm, scikit-learn)"
echo ""

# Install in order of importance
echo "Installing PyTorch..."
pip3 install torch torchvision torchaudio

echo "Installing PyTorch Geometric..."
pip3 install torch-geometric

echo "Installing e3nn (equivariant networks)..."
pip3 install e3nn

echo "Installing RDKit..."
pip3 install rdkit

echo "Installing scientific packages..."
pip3 install numpy scipy pandas

echo "Installing visualization..."
pip3 install matplotlib seaborn plotly

echo "Installing utilities..."
pip3 install scikit-learn tqdm

echo "Installing web interface..."
pip3 install gradio

echo ""
echo "✓ All dependencies installed!"
echo ""

# Step 4: Run Phase 2 evaluation
echo "Step 4/4: Running Phase 2 evaluation..."
echo "============================================"
echo ""

python3 evaluation/evaluate_phase2.py

echo ""
echo "============================================"
echo "✓ Setup Complete!"
echo "============================================"
echo ""
echo "🎉 Your ChemJEPA system is ready!"
echo ""
echo "Next steps:"
echo "  1. Review evaluation results above"
echo ""
echo "  2. Launch the web interface:"
echo "     ./launch.sh"
echo ""
echo "  3. Then open your browser to:"
echo "     http://localhost:7860"
echo ""
echo "Tip: Keep the terminal open while using the interface"
echo ""
