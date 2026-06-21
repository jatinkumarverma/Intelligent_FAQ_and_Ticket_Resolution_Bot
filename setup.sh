#!/bin/bash
# Setup script for Offline Support Bot on Linux/macOS

echo ""
echo "============================================"
echo "Offline Support Bot - Setup Script"
echo "============================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed"
    echo "Please install Python 3.9+ from https://www.python.org/"
    exit 1
fi

echo "[✓] Python found: $(python3 --version)"
echo ""

# Create virtual environment
echo "[*] Creating virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to create virtual environment"
    exit 1
fi
echo "[✓] Virtual environment created"
echo ""

# Activate virtual environment
echo "[*] Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to activate virtual environment"
    exit 1
fi
echo "[✓] Virtual environment activated"
echo ""

# Upgrade pip
echo "[*] Upgrading pip..."
python -m pip install --upgrade pip setuptools wheel > /dev/null 2>&1
echo "[✓] Pip upgraded"
echo ""

# Install requirements
echo "[*] Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install requirements"
    exit 1
fi
echo "[✓] Dependencies installed"
echo ""

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "[*] Creating .env file from template..."
    cp .env.example .env
    echo "[✓] .env file created (customize as needed)"
else
    echo "[✓] .env file already exists"
fi
echo ""

# Create necessary directories
echo "[*] Creating data directories..."
mkdir -p data/docs data/sample logs
echo "[✓] Directories created"
echo ""

echo "============================================"
echo "Setup Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "1. Install Ollama from https://ollama.ai/"
echo "2. Run: ollama pull mistral"
echo "3. Run: ollama serve (in a separate terminal)"
echo "4. Activate the virtual environment: source venv/bin/activate"
echo "5. Run the app: streamlit run ui/streamlit_app.py"
echo ""
echo "============================================"
