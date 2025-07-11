<div align="center">
  <img src="./assets/AI-ImagerForgeLogo.png" alt="AI-ImagerForge Logo" width="200"/>
  
  # AI-ImageForge
  
  *Universal AI Image Generation with Automatic GPU Optimization*
</div>

---

A professional-grade AI image generation application that automatically detects and optimizes for both NVIDIA and AMD GPUs. Features intelligent fallback systems, multiple AI models, and one-click deployment for competition-ready performance.

![AI-ImageForge](https://img.shields.io/badge/AI--ImageForge-v1.0-blue)
![GPU Support](https://img.shields.io/badge/GPU-NVIDIA%20%7C%20AMD-green)
![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red)

## ✨ Features

### 🖥️ Universal Hardware Support
- **Automatic GPU Detection**: Detects NVIDIA CUDA and AMD DirectML
- **Hardware Optimization**: Automatically configures optimal settings
- **CPU Fallback**: Works on any system without dedicated GPU
- **Memory Management**: Intelligent memory optimization for all hardware types

### 🤖 AI Generation Capabilities
- **Multiple AI Models**: Stable Diffusion v1.5, v2.1, and specialized models
- **Style-Specific Enhancement**: 6 distinct art styles with optimized prompts
- **Intelligent Fallbacks**: Graceful degradation when API models are unavailable
- **High-Quality Output**: 512×512 resolution with professional enhancement

### 🎨 Professional Interface
- **Real-time Hardware Monitoring**: Live system status and performance metrics
- **Advanced Controls**: Negative prompts, seeds, and quality modes
- **Example Gallery**: Pre-built prompts for quick experimentation
- **Progress Tracking**: Visual feedback during generation process

### 🚀 Performance Features
- **Batch Generation**: Multiple images with consistent quality
- **Performance Analytics**: Generation time and success rate tracking
- **Optimized Loading**: Smart model caching and memory management
- **Error Recovery**: Robust error handling with detailed diagnostics

## 🛠️ Installation

### One-Click Setup (Recommended)

1. **Download and Extract**: Extract all files to your desired directory
2. **Run Deployment**: Double-click `start.bat`
3. **Access Application**: Open browser to `http://localhost:8501`

The deployment script will automatically:
- ✅ Verify Python installation
- ✅ Create virtual environment
- ✅ Install all dependencies
- ✅ Detect hardware configuration
- ✅ Launch the application

### Manual Installation

```bash
# Clone or download the project
cd web-genAI

# Create virtual environment
python -m venv .venv

# Activate environment (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

## 🔧 System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, macOS 10.15+, Linux Ubuntu 18.04+
- **Python**: 3.8 or higher
- **RAM**: 4GB (8GB recommended)
- **Storage**: 2GB free space
- **Internet**: Required for AI model access

### Hardware Optimization

#### NVIDIA GPUs
- **Supported**: GeForce GTX/RTX series, Tesla, Quadro
- **Features**: CUDA acceleration, FP16 precision, memory optimization
- **Performance**: 3-10x faster generation than CPU

#### AMD GPUs
- **Supported**: Radeon RX series, Vega, RDNA/RDNA2
- **Features**: DirectML acceleration, optimized memory usage
- **Performance**: 2-5x faster generation than CPU

#### CPU Fallback
- **Supported**: Any x64 processor
- **Features**: Intelligent caching, optimized threading
- **Performance**: Reliable generation with demo fallbacks

## 🚀 Quick Start Guide

### 1. Get API Token
1. Visit [Hugging Face](https://huggingface.co/settings/tokens)
2. Create a new access token (free)
3. Copy the token for use in the application

### 2. Launch Application
- **Windows**: Double-click `start.bat`
- **Manual**: Run `streamlit run app.py`

### 3. Configure Settings
1. Enter your Hugging Face API token
2. Select your preferred art style
3. Choose quality mode based on your hardware

### 4. Generate Images
1. Enter a detailed description in English
2. Click "🚀 Generate Image"
3. Download your generated artwork

## 🎨 Art Styles

### 🏞️ Realistic
- **Best for**: Landscapes, portraits, photography-style images
- **Enhancement**: Photorealistic, highly detailed, professional photography
- **Example**: "A serene mountain lake at sunrise with perfect reflections"

### 🌸 Anime
- **Best for**: Character art, illustrations, manga-style content
- **Enhancement**: Anime style, manga style, high quality artwork
- **Example**: "Beautiful anime girl with flowing hair in cherry blossom garden"

### 🎨 Artistic
- **Best for**: Creative interpretations, abstract concepts
- **Enhancement**: Digital art, artistic masterpiece, trending on artstation
- **Example**: "Abstract representation of music with flowing colors"

### ✨ Fantasy
- **Best for**: Dragons, magical creatures, mystical scenes
- **Enhancement**: Fantasy art, magical, mystical, ethereal
- **Example**: "A majestic dragon with iridescent scales flying through storm clouds"

### 🌆 Cyberpunk
- **Best for**: Futuristic scenes, sci-fi environments
- **Enhancement**: Cyberpunk style, neon lights, futuristic, high tech
- **Example**: "Futuristic cityscape with neon lights and flying cars at night"

### 👤 Portrait
- **Best for**: Character portraits, detailed faces
- **Enhancement**: Portrait photography, professional lighting, high quality
- **Example**: "Elegant royal portrait with ornate crown and jewelry"

## 📊 Performance Optimization

### GPU Memory Management
```python
# Automatic optimization based on detected hardware
if gpu_type == 'nvidia':
    # CUDA optimization with FP16 precision
    torch.backends.cudnn.benchmark = True
    
elif gpu_type == 'amd':
    # DirectML optimization with memory management
    providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
```

### Model Fallback System
1. **Primary**: Stable Diffusion v1.5 (fastest, highest quality)
2. **Secondary**: Stable Diffusion v2.1 (backup for high load)
3. **Tertiary**: Alternative specialized models
4. **Fallback**: High-quality procedural generation

### Performance Tips
- **Use detailed prompts**: More specific descriptions yield better results
- **Choose appropriate style**: Match style to your desired output
- **Monitor system resources**: Check hardware status in sidebar
- **Batch generation**: Generate multiple variations efficiently

## 🛡️ Troubleshooting

### Common Issues and Solutions

#### start.bat closes automatically
**Problem**: The batch file starts but closes unexpectedly
**Solutions**:
1. Check Python installation - Ensure Python 3.8+ is installed and in PATH
2. Run as administrator - Right-click start.bat → "Run as administrator"
3. Check file location - Ensure you're in the correct directory with app.py

#### Python not found
**Problem**: "Python is not installed or not in PATH"
**Solutions**:
1. Download Python from https://www.python.org/downloads/
2. During installation, check "Add Python to PATH"
3. Restart command prompt/terminal
4. Test with: `python --version`

#### Virtual environment issues
**Problem**: Virtual environment creation or activation fails
**Solutions**:
1. Delete `.venv` folder and try again
2. Run: `python -m venv .venv` manually
3. Check disk space (need at least 2GB free)
4. Try running from a folder without spaces in the name

#### Package installation failures
**Problem**: pip install commands fail
**Solutions**:
1. Check internet connection
2. Update pip: `python -m pip install --upgrade pip`
3. Try with administrator privileges
4. Use alternative installation: `python launcher.py`

#### Streamlit won't start
**Problem**: Application fails to launch or crashes
**Solutions**:
1. Check if port 8501 is available
2. Try different port: `streamlit run app.py --server.port 8502`
3. Clear Streamlit cache: Delete `.streamlit` folder
4. Check firewall/antivirus settings

#### GPU not detected
**Problem**: System shows "CPU execution only"
**Solutions**:
1. **For NVIDIA**: Install latest GPU drivers from nvidia.com
2. **For AMD**: Ensure DirectML is installed with Windows updates
3. Restart computer after driver installation

### Alternative Launch Methods

If start.bat fails, try these alternatives:

1. **Cross-Platform Launcher**:
   ```bash
   python launcher.py
   ```

2. **Manual Streamlit**:
   ```bash
   .venv\Scripts\activate
   streamlit run app.py
   ```

3. **Different Port**:
   ```bash
   streamlit run app.py --server.port 8502
   ```

## 📁 Project Structure

This project includes a comprehensive `.gitignore` file that excludes:

- **Virtual environments** (`.venv/`, `venv/`)
- **Python cache files** (`__pycache__/`, `*.pyc`)
- **AI model caches** (`.cache/huggingface/`, `.cache/torch/`)
- **Generated images** (`*.png`, `*.jpg`, etc.)
- **IDE files** (`.vscode/`, `.idea/`)
- **Logs and temporary files**
- **API keys and secrets** (security)

This ensures only essential source code is tracked in version control.

---

**🚀 Ready to create amazing AI-generated images with professional quality and universal hardware support!**
