#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "======================================"
echo -e "${BLUE}PyPotteryInk Installation${NC}"
echo "======================================"
echo

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} Python3 is not installed"
    echo "Please install Python 3.8+ from https://www.python.org"
    exit 1
fi

echo -e "${GREEN}[INFO]${NC} Python found:"
python3 --version

# Detect OS
OS="$(uname -s)"
case "${OS}" in
    Linux*)     OS_NAME="Linux";;
    Darwin*)    OS_NAME="macOS";;
    *)          OS_NAME="Unknown";;
esac
echo -e "${GREEN}[INFO]${NC} Detected OS: ${OS_NAME}"

# Create virtual environment
echo
echo -e "${GREEN}[INFO]${NC} Creating virtual environment..."
python3 -m venv .venv

# Activate virtual environment
echo -e "${GREEN}[INFO]${NC} Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo
echo -e "${GREEN}[INFO]${NC} Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo
echo -e "${GREEN}[INFO]${NC} Installing requirements..."
pip install -r requirements.txt

# Create models directory
echo
echo -e "${GREEN}[INFO]${NC} Creating models directory..."
mkdir -p models

# Create run script
echo
echo -e "${GREEN}[INFO]${NC} Creating run script..."
cat > run_app.sh << 'EOF'
#!/bin/bash
echo "Starting PyPotteryInk Gradio Interface..."
source .venv/bin/activate
python app.py
EOF
chmod +x run_app.sh

# Check GPU availability
echo
echo -e "${GREEN}[INFO]${NC} Checking GPU availability..."
if [[ "$OS_NAME" == "macOS" ]]; then
    if python3 -c "import torch; exit(0 if torch.backends.mps.is_available() else 1)" 2>/dev/null; then
        echo -e "${GREEN}[INFO]${NC} Metal Performance Shaders (MPS) available for GPU acceleration"
    else
        echo -e "${YELLOW}[WARNING]${NC} MPS not available, will use CPU mode"
    fi
else
    if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        echo -e "${GREEN}[INFO]${NC} CUDA available for GPU acceleration"
    else
        echo -e "${YELLOW}[WARNING]${NC} CUDA not available, will use CPU mode"
    fi
fi

# Success message
echo
echo "======================================"
echo -e "${GREEN}Installation completed successfully!${NC}"
echo "======================================"
echo
echo "To run the application:"
echo -e "  ${BLUE}./run_app.sh${NC}"
echo
echo "The interface will open in your browser at http://localhost:7860"
echo