#!/bin/bash
# AI-ImageForge - Cross-Platform Launcher
# Universal support for Linux and macOS
# ==========================================

set -e

echo ""
echo "==============================================="
echo "  🎨 AI-ImageForge - Cross-Platform Setup"
echo "==============================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check Python installation
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        echo -e "${RED}❌ Python is not installed or not in PATH${NC}"
        echo ""
        echo "Please install Python 3.9+ from:"
        echo "https://www.python.org/downloads/"
        echo ""
        exit 1
    fi
    
    echo -e "${GREEN}✅ Python detected${NC}"
    $PYTHON_CMD --version
    if ! $PYTHON_CMD -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)"; then
        echo -e "${RED}❌ Python 3.9+ is required${NC}"
        exit 1
    fi
}

# Setup virtual environment
setup_venv() {
    echo ""
    echo -e "${BLUE}📦 Setting up virtual environment...${NC}"
    
    if [ ! -d ".venv" ]; then
        echo "Creating new virtual environment..."
        $PYTHON_CMD -m venv .venv
    fi
    
    echo "Activating virtual environment..."
    source .venv/bin/activate
    
    echo -e "${GREEN}✅ Virtual environment ready${NC}"
}

# Install dependencies
install_dependencies() {
    echo ""
    echo -e "${BLUE}🔧 Installing dependencies...${NC}"
    
    # Upgrade pip first
    echo "Upgrading pip..."
    pip install --upgrade pip
    
    # Install requirements
    echo "Installing AI-ImageForge dependencies..."
    if pip install -r requirements.txt; then
        echo -e "${GREEN}✅ All dependencies installed successfully${NC}"
    else
        echo -e "${YELLOW}⚠️ Some packages failed. Trying alternative installation...${NC}"
        install_core_packages
    fi
}

# Install core packages individually
install_core_packages() {
    echo "Installing core packages..."
    
    # Core packages
    pip install "streamlit>=1.30.0"
    pip install "Pillow>=10.4.0"
    pip install "requests>=2.28.0"
    pip install "huggingface-hub>=1.0.0"
    pip install "onnxruntime>=1.15.0"
}

# Detect hardware
detect_hardware() {
    echo ""
    echo -e "${BLUE}🖥️ Detecting hardware configuration...${NC}"
    
    $PYTHON_CMD -c "
import platform
print(f'OS: {platform.system()} {platform.release()}')
print(f'Architecture: {platform.architecture()[0]}')

try:
    import torch
    if torch.cuda.is_available():
        print('✅ NVIDIA CUDA detected')
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    else:
        print('⚠️ NVIDIA CUDA not available')
except ImportError:
    print('⚠️ PyTorch not available')

try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    if 'CUDAExecutionProvider' in providers:
        print('✅ NVIDIA CUDA via ONNX detected')
    else:
        print('ℹ️ CPU execution only')
    print(f'Available providers: {providers}')
except ImportError:
    print('⚠️ ONNX Runtime not available')
"
}

# Setup environment variables
setup_environment() {
    export PYTHONWARNINGS=ignore
    export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4
}

# Launch application
launch_app() {
    echo ""
    echo -e "${BLUE}🌐 Starting AI-ImageForge...${NC}"
    echo ""
    echo "Open your browser and navigate to:"
    echo -e "${GREEN}http://localhost:8501${NC}"
    echo ""
    echo "Press Ctrl+C to stop the application"
    echo ""
    
    streamlit run app.py --server.port 8501 --server.address localhost --theme.primaryColor "#FF6B6B"
}

# Cleanup function
cleanup() {
    echo ""
    echo -e "${GREEN}Application stopped.${NC}"
    deactivate 2>/dev/null || true
}

# Set trap for cleanup
trap cleanup EXIT

# Main execution
main() {
    check_python
    setup_venv
    install_dependencies
    setup_environment
    detect_hardware
    
    echo ""
    echo "==============================================="
    launch_app
}

# Run main function
main
