@echo off
setlocal EnableDelayedExpansion
REM AI Image Generator - Fixed One-Click Deployment
REM Universal support for NVIDIA and AMD GPUs
REM ===========================================

title AI Image Generator Deployment

echo.
echo ===============================================
echo   🚀 AI Image Generator - One-Click Setup
echo ===============================================
echo.

REM Check Python installation
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo.
    echo Please install Python 3.8+ from:
    echo https://www.python.org/downloads/
    echo.
    echo Press any key to exit...
    pause >nul
    exit /b 1
)

echo ✅ Python detected
python --version

echo.
echo 📦 Setting up virtual environment...

REM Create virtual environment if it doesn't exist
if not exist ".venv" (
    echo Creating new virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ❌ Failed to create virtual environment
        goto :error_handler
    )
    echo ✅ Virtual environment created
) else (
    echo ✅ Virtual environment found
)

REM Activate virtual environment
echo Activating virtual environment...
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    echo ✅ Virtual environment activated
) else (
    echo ❌ Virtual environment activation script not found
    goto :error_handler
)

echo.
echo 🔧 Installing dependencies...

REM Upgrade pip first
echo Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1

REM Install requirements
echo Installing AI Image Generator dependencies...
echo This may take a few minutes, please wait...
pip install -r requirements.txt >nul 2>&1
if errorlevel 1 (
    echo ⚠️ Some packages failed. Installing core packages...
    pip install "streamlit>=1.30.0" >nul 2>&1
    pip install "Pillow>=10.4.0" >nul 2>&1
    pip install "requests>=2.28.0" >nul 2>&1
    pip install "numpy>=1.24.0" >nul 2>&1
    echo ✅ Core packages installed
) else (
    echo ✅ All dependencies installed successfully
)

echo.
echo 🖥️ Detecting hardware configuration...
python -c "import platform; print(f'OS: {platform.system()} {platform.release()}')" 2>nul

echo.
echo 🌐 Starting AI Image Generator...
echo.
echo Open your browser and navigate to:
echo http://localhost:8501
echo.
echo Press Ctrl+C to stop the application
echo.

REM Check if app.py exists
if not exist "app.py" (
    echo ❌ app.py not found in current directory
    echo Current directory: %CD%
    goto :error_handler
)

REM Set environment variables
set PYTHONWARNINGS=ignore
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

REM Start Streamlit application
streamlit run app.py --server.port 8501 --server.address localhost --theme.primaryColor="#FF6B6B"

goto :normal_exit

:error_handler
echo.
echo ===============================================
echo ❌ Setup encountered errors
echo ===============================================
echo.
echo Please try the following:
echo 1. Ensure Python 3.8+ is installed
echo 2. Check your internet connection
echo 3. Try running as administrator
echo 4. Use test.bat to diagnose issues
echo.
echo Press any key to exit...
pause >nul
exit /b 1

:normal_exit
echo.
echo ===============================================
echo ✅ Application stopped normally
echo ===============================================
echo.
echo Press any key to exit...
pause >nul
