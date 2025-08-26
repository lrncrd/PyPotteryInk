#!/bin/bash

echo "============================================"
echo "PyPotteryInk Installation Script for Unix/Linux/macOS"
echo "============================================"
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_error() {
    echo -e "${RED}ERROR: $1${NC}"
}

print_success() {
    echo -e "${GREEN}$1${NC}"
}

print_warning() {
    echo -e "${YELLOW}$1${NC}"
}

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed."
    echo "Please install Python 3.10 or higher:"
    echo "  - macOS: brew install python3"
    echo "  - Ubuntu/Debian: sudo apt-get install python3 python3-pip python3-venv"
    echo "  - Fedora: sudo dnf install python3 python3-pip"
    exit 1
fi

print_success "Python 3 is installed."
echo

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Found Python version: $PYTHON_VERSION"
echo

# Check if venv is available
if ! python3 -m venv --help &> /dev/null; then
    print_error "Python venv module is not installed."
    echo "Please install it:"
    echo "  - Ubuntu/Debian: sudo apt-get install python3-venv"
    echo "  - Fedora: sudo dnf install python3-venv"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    print_error "Failed to create virtual environment."
    exit 1
fi
print_success "Virtual environment created successfully."
echo

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    print_error "Failed to activate virtual environment."
    exit 1
fi
print_success "Virtual environment activated."
echo

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip
echo

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    print_error "Failed to install requirements."
    echo "Please check your internet connection and try again."
    exit 1
fi
print_success "Requirements installed successfully."
echo

# Install Gradio
echo "Installing Gradio..."
pip install gradio>=4.0.0
if [ $? -ne 0 ]; then
    print_error "Failed to install Gradio."
    exit 1
fi
print_success "Gradio installed successfully."
echo

# Create models directory
echo "Creating models directory..."
mkdir -p models
echo

# Create run scripts
echo "Creating run scripts..."

# Create run_app.sh
cat > run_app.sh << 'EOF'
#!/bin/bash
echo "Starting PyPotteryInk Gradio Interface..."
source venv/bin/activate
python app.py
EOF
chmod +x run_app.sh

# Create run_test.sh
cat > run_test.sh << 'EOF'
#!/bin/bash
echo "Running PyPotteryInk Tests..."
source venv/bin/activate
python test.py
EOF
chmod +x run_test.sh

print_success "Run scripts created."
echo

echo "============================================"
print_success "Installation completed successfully!"
echo "============================================"
echo
echo "To run the Gradio interface:"
echo "  - Run: ./run_app.sh"
echo "  - Open browser at http://localhost:7860"
echo
echo "To run tests:"
echo "  - Run: ./run_test.sh"
echo
echo "To activate the virtual environment manually:"
echo "  - Run: source venv/bin/activate"
echo