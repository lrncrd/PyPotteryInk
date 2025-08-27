@echo off
echo ======================================
echo PyPotteryInk Installation for Windows
echo ======================================
echo.

REM Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org
    pause
    exit /b 1
)

echo [INFO] Python found:
python --version

REM Create virtual environment
echo.
echo [INFO] Creating virtual environment...
python -m venv .venv

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call .venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo.
echo [INFO] Installing requirements...
pip install -r requirements.txt

REM Create models directory
echo.
echo [INFO] Creating models directory...
if not exist "models" mkdir models

REM Create run script
echo.
echo [INFO] Creating run script...
echo @echo off > run_app.bat
echo echo Starting PyPotteryInk Gradio Interface... >> run_app.bat
echo call .venv\Scripts\activate.bat >> run_app.bat
echo python app.py >> run_app.bat

REM Success message
echo.
echo ======================================
echo Installation completed successfully!
echo ======================================
echo.
echo To run the application:
echo   1. Double-click run_app.bat
echo   OR
echo   2. Open Command Prompt and run: run_app.bat
echo.
echo The interface will open in your browser at http://localhost:7860
echo.
pause