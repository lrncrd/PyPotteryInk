@echo off
echo.
echo ================================================================================
echo                          PyPotteryInk Setup
echo ================================================================================
echo.

:: Check CUDA availability through nvidia-smi
nvidia-smi >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [✓] NVIDIA GPU detected
    for /f "tokens=2 delims=," %%a in ('nvidia-smi --query-gpu^=name --format^=csv ^| findstr /v "name"') do set GPU_NAME=%%a
    echo     • GPU: %GPU_NAME%
    
    :: Get CUDA version
    for /f "tokens=3" %%a in ('nvidia-smi ^| findstr "CUDA Version"') do set CUDA_VERSION=%%a
    echo     • CUDA Version: %CUDA_VERSION%
    set CUDA_AVAILABLE=1
) else (
    echo [!] No NVIDIA GPU detected - will install CPU-only version
    set CUDA_AVAILABLE=0
)

echo.
echo Checking Python environment...

:: Check if Python virtual environment exists
set VENV_EXISTS=0
if exist "venv" (
    echo [✓] Virtual environment already exists
    set VENV_EXISTS=1
) else (
    echo [*] Creating virtual environment...
    python -m venv venv
    if %ERRORLEVEL% NEQ 0 (
        echo [!] Failed to create virtual environment. Please ensure Python is installed.
        pause
        exit /b 1
    )
    set VENV_EXISTS=0
)

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Only install packages if venv is newly created
if %VENV_EXISTS%==0 (
    echo [*] New virtual environment detected, installing packages...
    
    :: Update pip first
    python -m pip install --upgrade pip
    if %ERRORLEVEL% NEQ 0 (
        echo [!] Failed to upgrade pip.
        pause
        exit /b 1
    )
    
    :: Install PyTorch based on CUDA availability
    if %CUDA_AVAILABLE%==1 (
        echo [*] Installing PyTorch with CUDA support...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ) else (
        echo [*] Installing CPU-only PyTorch...
        pip install torch torchvision torchaudio
    )
    
    if %ERRORLEVEL% NEQ 0 (
        echo [!] Failed to install PyTorch.
        pause
        exit /b 1
    )
    
    :: Install requirements from requirements.txt
    if exist "requirements.txt" (
        echo [*] Installing packages from requirements.txt...
        pip install -r requirements.txt
        
        if %ERRORLEVEL% NEQ 0 (
            echo [!] Failed to install requirements from requirements.txt.
            pause
            exit /b 1
        )
    ) else (
        echo [!] requirements.txt not found. Installing basic packages...
        pip install gradio huggingface-hub transformers diffusers peft numpy opencv-python matplotlib scikit-image seaborn pillow requests
        
        if %ERRORLEVEL% NEQ 0 (
            echo [!] Failed to install requirements.
            pause
            exit /b 1
        )
    )
)

:: Create necessary directories
echo [*] Creating directories...
if not exist "models" mkdir models
if not exist "temp_input" mkdir temp_input
if not exist "temp_output" mkdir temp_output
if not exist "temp_diagnostics" mkdir temp_diagnostics

:: Model directories prepared (models are downloaded via the app)
echo [*] Model directories prepared - models will be downloaded automatically when needed

:: Verify installation
echo.
echo Verifying PyTorch installation...
python -c "import torch; print(f'[✓] PyTorch {torch.__version__}'); print(f'[✓] CUDA available: {torch.cuda.is_available()}'); print(f'[✓] GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

if %ERRORLEVEL% NEQ 0 (
    echo [!] Failed to verify PyTorch installation.
    pause
    exit /b 1
)

:: Verify other packages
echo Verifying other packages...
python -c "import gradio, transformers, diffusers; print('[✓] All packages installed successfully')"

if %ERRORLEVEL% NEQ 0 (
    echo [!] Failed to verify package installation.
    pause
    exit /b 1
)

:: Start the application
echo.
echo [*] Starting PyPotteryInk...
python app.py

:: Keep the window open if there's an error
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [!] An error occurred while starting PyPotteryInk. Please check the messages above.
    pause
    exit /b 1
)

pause
