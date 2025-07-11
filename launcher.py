#!/usr/bin/env python3
"""
AI-ImageForge Launcher
Universal cross-platform launcher with automatic environment setup
"""

import os
import sys
import subprocess
import platform
import venv
from pathlib import Path

class AIGeneratorLauncher:
    def __init__(self):
        self.project_dir = Path(__file__).parent
        self.venv_dir = self.project_dir / ".venv"
        self.requirements_file = self.project_dir / "requirements.txt"
        self.app_file = self.project_dir / "app.py"
        
    def print_header(self):
        """Print application header"""
        print("\n" + "="*50)
        print("🎨 AI-ImageForge - Universal Launcher")
        print("="*50)
        print(f"Platform: {platform.system()} {platform.release()}")
        print(f"Python: {sys.version.split()[0]}")
        print(f"Architecture: {platform.architecture()[0]}")
        print("="*50 + "\n")
    
    def check_python_version(self):
        """Check if Python version is compatible"""
        if sys.version_info < (3, 8):
            print("❌ Python 3.8+ is required")
            print(f"Current version: {sys.version}")
            return False
        print(f"✅ Python {sys.version.split()[0]} detected")
        return True
    
    def create_virtual_environment(self):
        """Create virtual environment if it doesn't exist"""
        if not self.venv_dir.exists():
            print("📦 Creating virtual environment...")
            venv.create(self.venv_dir, with_pip=True)
            print("✅ Virtual environment created")
        else:
            print("✅ Virtual environment found")
    
    def get_pip_command(self):
        """Get pip command for current platform"""
        if platform.system() == "Windows":
            return str(self.venv_dir / "Scripts" / "pip.exe")
        else:
            return str(self.venv_dir / "bin" / "pip")
    
    def get_python_command(self):
        """Get Python command for current platform"""
        if platform.system() == "Windows":
            return str(self.venv_dir / "Scripts" / "python.exe")
        else:
            return str(self.venv_dir / "bin" / "python")
    
    def install_requirements(self):
        """Install required packages"""
        if not self.requirements_file.exists():
            print("❌ requirements.txt not found")
            return False
        
        print("🔧 Installing dependencies...")
        pip_cmd = self.get_pip_command()
        
        # Upgrade pip first
        subprocess.run([pip_cmd, "install", "--upgrade", "pip"], check=False)
        
        # Install requirements
        result = subprocess.run([
            pip_cmd, "install", "-r", str(self.requirements_file)
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print("⚠️ Some packages failed to install. Trying alternative installation...")
            self.install_core_packages()
        else:
            print("✅ All dependencies installed successfully")
        
        return True
    
    def install_core_packages(self):
        """Install core packages individually"""
        pip_cmd = self.get_pip_command()
        core_packages = [
            "streamlit>=1.30.0",
            "Pillow>=10.4.0",
            "requests>=2.28.0",
            "numpy>=1.24.0"
        ]
        
        for package in core_packages:
            print(f"Installing {package}...")
            subprocess.run([pip_cmd, "install", package], check=False)
        
        # Try PyTorch
        print("Installing PyTorch...")
        subprocess.run([
            pip_cmd, "install", "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cpu"
        ], check=False)
        
        # Try Hugging Face packages
        hf_packages = ["transformers", "diffusers", "accelerate", "huggingface-hub"]
        for package in hf_packages:
            print(f"Installing {package}...")
            subprocess.run([pip_cmd, "install", package], check=False)
        
        # Try ONNX for AMD support
        print("Installing ONNX Runtime...")
        subprocess.run([pip_cmd, "install", "onnxruntime"], check=False)
        if platform.system() == "Windows":
            subprocess.run([pip_cmd, "install", "onnxruntime-directml"], check=False)
    
    def detect_hardware(self):
        """Detect and display hardware information"""
        print("🖥️ Detecting hardware configuration...")
        python_cmd = self.get_python_command()
        
        detection_script = """
import platform
print(f'OS: {platform.system()} {platform.release()}')
print(f'Architecture: {platform.architecture()[0]}')

try:
    import torch
    if torch.cuda.is_available():
        print('✅ NVIDIA CUDA detected')
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'CUDA Version: {torch.version.cuda}')
    else:
        print('⚠️ NVIDIA CUDA not available')
except ImportError:
    print('⚠️ PyTorch not available')

try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    if 'DmlExecutionProvider' in providers:
        print('✅ AMD DirectML detected')
    elif 'CUDAExecutionProvider' in providers:
        print('✅ NVIDIA CUDA via ONNX detected')
    else:
        print('ℹ️ CPU execution only')
    print(f'Available providers: {", ".join(providers)}')
except ImportError:
    print('⚠️ ONNX Runtime not available')
"""
        
        subprocess.run([python_cmd, "-c", detection_script], check=False)
    
    def setup_environment(self):
        """Setup environment variables"""
        env_vars = {
            'PYTHONWARNINGS': 'ignore',
            'STREAMLIT_BROWSER_GATHER_USAGE_STATS': 'false',
            'OMP_NUM_THREADS': '4',
            'MKL_NUM_THREADS': '4'
        }
        
        for key, value in env_vars.items():
            os.environ[key] = value
    
    def launch_application(self):
        """Launch the Streamlit application"""
        if not self.app_file.exists():
            print("❌ app.py not found")
            return False
        
        print("🌐 Starting AI-ImageForge...")
        print("\nOpen your browser and navigate to:")
        print("http://localhost:8501")
        print("\nPress Ctrl+C to stop the application\n")
        
        python_cmd = self.get_python_command()
        
        try:
            subprocess.run([
                python_cmd, "-m", "streamlit", "run", str(self.app_file),
                "--server.port", "8501",
                "--server.address", "localhost",
                "--theme.primaryColor", "#FF6B6B"
            ], check=True)
        except KeyboardInterrupt:
            print("\n\n✅ Application stopped by user")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Application failed to start: {e}")
            return False
        
        return True
    
    def run(self):
        """Main launcher function"""
        try:
            self.print_header()
            
            if not self.check_python_version():
                return False
            
            self.create_virtual_environment()
            self.install_requirements()
            self.setup_environment()
            self.detect_hardware()
            
            print("\n" + "="*50)
            return self.launch_application()
            
        except Exception as e:
            print(f"\n❌ Launcher error: {e}")
            return False

if __name__ == "__main__":
    launcher = AIGeneratorLauncher()
    success = launcher.run()
    
    if not success:
        print("\n⚠️ Setup incomplete. Please check the errors above.")
        if platform.system() == "Windows":
            input("Press Enter to exit...")
        sys.exit(1)
