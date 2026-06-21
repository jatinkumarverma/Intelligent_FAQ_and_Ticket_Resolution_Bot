@echo off
REM Setup script for Offline Support Bot on Windows
REM This script sets up the Python environment and dependencies

echo.
echo ============================================
echo Offline Support Bot - Setup Script
echo ============================================
echo.

@REM REM Check if Python is installed
@REM python --version >nul 2>&1
@REM if errorlevel 1 (
@REM     echo [ERROR] Python is not installed or not in PATH
@REM     echo Please install Python 3.9+ from https://www.python.org/
@REM     exit /b 1
@REM )

@REM echo [✓] Python found
@REM echo.

@REM REM Create virtual environment
@REM echo [*] Creating virtual environment...
@REM python -m venv venv311
@REM if errorlevel 1 (
@REM     echo [ERROR] Failed to create virtual environment
@REM     exit /b 1
@REM )
@REM echo [✓] Virtual environment created
@REM echo.

@REM REM Activate virtual environment
@REM echo [*] Activating virtual environment...
@REM call venv311\Scripts\activate.bat
@REM if errorlevel 1 (
@REM     echo [ERROR] Failed to activate virtual environment
@REM     exit /b 1
@REM )
@REM echo [✓] Virtual environment activated
@REM echo.

REM Upgrade pip
echo [*] Upgrading pip...
python -m pip install --upgrade pip setuptools wheel >nul 2>&1
echo [✓] Pip upgraded
echo.

REM Install requirements
echo [*] Installing dependencies (this may take a few minutes)...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install requirements
    exit /b 1
)
echo [✓] Dependencies installed
echo.

REM Create .env file if it doesn't exist
if not exist .env (
    echo [*] Creating .env file from template...
    copy .env.example .env >nul
    echo [✓] .env file created (customize as needed)
) else (
    echo [✓] .env file already exists
)
echo.

REM Create necessary directories
echo [*] Creating data directories...
if not exist data\docs mkdir data\docs
if not exist data\sample mkdir data\sample
if not exist logs mkdir logs
echo [✓] Directories created
echo.

echo ============================================
echo Setup Complete!
echo ============================================
echo.
echo Next steps:
echo 1. Install Ollama from https://ollama.ai/
echo 2. Run: ollama pull mistral
echo 3. Run: ollama serve (in a separate terminal)
echo 4. Activate the virtual environment: venv311\Scripts\activate.bat
echo 5. Run the app: streamlit run ui/streamlit_app.py
echo.
echo ============================================
pause
