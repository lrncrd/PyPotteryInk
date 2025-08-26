@echo off
echo ============================================
echo PyPotteryInk Installation Script for Windows
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.10 or higher from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

echo Python is installed.
echo.

REM Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Found Python version: %PYTHON_VERSION%
echo.

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment.
    echo Make sure you have the venv module installed.
    pause
    exit /b 1
)
echo Virtual environment created successfully.
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment.
    pause
    exit /b 1
)
echo Virtual environment activated.
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install requirements.
    echo Please check your internet connection and try again.
    pause
    exit /b 1
)
echo Requirements installed successfully.
echo.

REM Install Gradio
echo Installing Gradio...
pip install gradio>=4.0.0
if errorlevel 1 (
    echo ERROR: Failed to install Gradio.
    pause
    exit /b 1
)
echo Gradio installed successfully.
echo.

REM Create models directory
echo Creating models directory...
if not exist models mkdir models
echo.

REM Create run scripts
echo Creating run scripts...

REM Create run_app.bat
echo @echo off > run_app.bat
echo echo Starting PyPotteryInk Gradio Interface... >> run_app.bat
echo call venv\Scripts\activate.bat >> run_app.bat
echo python app.py >> run_app.bat
echo pause >> run_app.bat

REM Create run_test.bat
echo @echo off > run_test.bat
echo echo Running PyPotteryInk Tests... >> run_test.bat
echo call venv\Scripts\activate.bat >> run_test.bat
echo python test.py >> run_test.bat
echo pause >> run_test.bat

echo Run scripts created.
echo.

echo ============================================
echo Installation completed successfully!
echo ============================================
echo.
echo To run the Gradio interface:
echo   - Double-click run_app.bat or run: run_app.bat
echo   - Open browser at http://localhost:7860
echo.
echo To run tests:
echo   - Double-click run_test.bat or run: run_test.bat
echo.
echo To activate the virtual environment manually:
echo   - Run: venv\Scripts\activate.bat
echo.
pause