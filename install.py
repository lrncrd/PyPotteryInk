#!/usr/bin/env python3
"""
Unified installation script for PyPotteryInk
Works on Windows, macOS, and Linux
"""

import os
import sys
import subprocess
import platform
import shutil


def get_python_command():
    """Get the correct Python command for the system"""
    # Use the Python that's running this script
    return sys.executable


def create_venv():
    """Create virtual environment"""
    python_cmd = get_python_command()
    print("🔧 Creating virtual environment...")
    
    # Check if we're running from within the venv we're trying to delete
    if os.path.exists(".venv"):
        venv_python = os.path.join(".venv", "Scripts", "python.exe") if platform.system() == "Windows" else os.path.join(".venv", "bin", "python")
        
        # Check if running from within the venv
        try:
            if os.path.exists(venv_python) and os.path.samefile(sys.executable, venv_python):
                print("⚠️  Warning: Running from within the virtual environment that needs to be recreated.")
                print("   Please run this script from outside the virtual environment:")
                print(f"   deactivate")
                print(f"   python3 install.py")
                sys.exit(1)
        except:
            # If samefile fails, check if .venv is in the path
            if ".venv" in sys.executable:
                print("⚠️  Warning: It appears you're running from within the virtual environment.")
                print("   Please run this script from outside the virtual environment:")
                print(f"   deactivate")
                print(f"   python3 install.py")
                sys.exit(1)
        
        print("   Removing existing virtual environment...")
        shutil.rmtree(".venv", ignore_errors=True)
    
    # Create new venv
    subprocess.run([python_cmd, "-m", "venv", ".venv"], check=True)
    print("✅ Virtual environment created")


def get_pip_command():
    """Get the correct pip command based on OS"""
    if platform.system() == "Windows":
        return os.path.join(".venv", "Scripts", "pip")
    else:
        return os.path.join(".venv", "bin", "pip")


def get_activate_command():
    """Get the correct activation command based on OS"""
    if platform.system() == "Windows":
        return os.path.join(".venv", "Scripts", "activate.bat")
    else:
        return f"source {os.path.join('.venv', 'bin', 'activate')}"


def install_dependencies():
    """Install all required dependencies"""
    pip_cmd = get_pip_command()
    print("\n📦 Installing dependencies...")
    
    # Upgrade pip first
    print("   Upgrading pip...")
    subprocess.run([pip_cmd, "install", "--upgrade", "pip"], check=True)
    
    # Install requirements
    print("   Installing requirements...")
    subprocess.run([pip_cmd, "install", "-r", "requirements.txt"], check=True)
    print("✅ Dependencies installed")


def download_models():
    """Download required models"""
    print("\n🤖 Downloading models...")
    python_cmd = os.path.join(".venv", "Scripts", "python") if platform.system() == "Windows" else os.path.join(".venv", "bin", "python")
    
    # Create a simple script to download models
    download_script = '''
import os
from huggingface_hub import hf_hub_download

# Create models directory
os.makedirs("models", exist_ok=True)

# Download 10k model
print("   Downloading 10k model (38.3MB)...")
try:
    hf_hub_download(
        repo_id="lrncrd/PyPotteryInk",
        filename="model_10k.pkl",
        local_dir="models",
        local_dir_use_symlinks=False
    )
    print("   ✓ 10k model downloaded")
except Exception as e:
    print(f"   ⚠️  Could not download 10k model: {e}")
    print("   You can download it manually from: https://huggingface.co/lrncrd/PyPotteryInk")
'''
    
    # Run the download script
    subprocess.run([python_cmd, "-c", download_script], check=True)
    print("✅ Model download complete")


def create_run_script():
    """Create platform-specific run script"""
    print("\n📝 Creating run script...")
    
    if platform.system() == "Windows":
        script_name = "run.bat"
        script_content = '''@echo off
echo Starting PyPotteryInk...
call .venv\\Scripts\\activate
echo.
echo Checking CUDA availability...
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
echo.
python app.py
pause
'''
        # Also create a debug version for Windows
        debug_script = '''@echo off
echo Starting PyPotteryInk in DEBUG mode...
call .venv\\Scripts\\activate
set PYTHONUNBUFFERED=1
echo.
echo Debug information:
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
echo.
python app.py
pause
'''
        with open("run_debug.bat", "w") as f:
            f.write(debug_script)
        print("   Also created: run_debug.bat (for troubleshooting)")
        
    else:
        script_name = "run.sh"
        script_content = '''#!/bin/bash
echo "Starting PyPotteryInk..."
source .venv/bin/activate
echo ""
echo "Checking GPU availability..."
python -c "import torch; device = 'MPS' if torch.backends.mps.is_available() else 'CUDA' if torch.cuda.is_available() else 'CPU'; print(f'Device: {device}')"
echo ""
python app.py
'''
    
    with open(script_name, "w") as f:
        f.write(script_content)
    
    # Make it executable on Unix-like systems
    if platform.system() != "Windows":
        os.chmod(script_name, 0o755)
    
    print(f"✅ Run script created: {script_name}")
    return script_name


def detect_gpu():
    """Detect available GPU"""
    python_cmd = os.path.join(".venv", "Scripts", "python") if platform.system() == "Windows" else os.path.join(".venv", "bin", "python")
    
    detect_script = '''
import torch
if torch.cuda.is_available():
    print(f"CUDA GPU detected: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    print("Apple Silicon GPU detected")
else:
    print("No GPU detected - will use CPU")
'''
    
    print("\n🔍 Detecting GPU...")
    subprocess.run([python_cmd, "-c", detect_script])


def main():
    """Main installation process"""
    print("=================================")
    print("PyPotteryInk Installation")
    print("=================================")
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Python: {sys.version.split()[0]}")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("\n❌ Error: Python 3.8 or higher is required")
        sys.exit(1)
    
    try:
        # Run installation steps
        create_venv()
        install_dependencies()
        download_models()
        run_script = create_run_script()
        detect_gpu()
        
        # Success message
        print("\n" + "="*50)
        print("✅ Installation completed successfully!")
        print("="*50)
        print("\nTo run PyPotteryInk:")
        if platform.system() == "Windows":
            print(f"  Double-click {run_script} or run it from terminal")
        else:
            print(f"  Run: ./{run_script}")
        print("\nThe web interface will open automatically in your browser.")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Installation failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()