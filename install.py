#!/usr/bin/env python3
"""
Unified installation script for PyPotteryInk
Works on Windows, macOS, and Linux
Enhanced GPU detection and PyTorch installation
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


def check_gpu_support():
    """Check what GPU support is available on the system"""
    print("\n🔍 Checking system GPU support...")
    
    # Check for NVIDIA GPU
    nvidia_gpu = False
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            nvidia_gpu = True
            print("   ✅ NVIDIA GPU detected")
            # Extract GPU info
            for line in result.stdout.split('\n'):
                if 'GeForce' in line or 'RTX' in line or 'GTX' in line or 'Quadro' in line:
                    gpu_name = line.strip()
                    if '|' in gpu_name:
                        gpu_name = gpu_name.split('|')[1].strip()
                    print(f"   GPU: {gpu_name}")
                    break
        else:
            print("   ⚠️  No NVIDIA GPU or drivers not installed")
    except FileNotFoundError:
        print("   ⚠️  nvidia-smi not found - no NVIDIA GPU detected")
    
    return nvidia_gpu


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
                print(f"   python install.py")
                sys.exit(1)
        except:
            # If samefile fails, check if .venv is in the path
            if ".venv" in sys.executable:
                print("⚠️  Warning: It appears you're running from within the virtual environment.")
                print("   Please run this script from outside the virtual environment:")
                print(f"   deactivate")
                print(f"   python install.py")
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


def install_pytorch_with_cuda(pip_cmd, has_nvidia_gpu):
    """Install PyTorch with appropriate CUDA support"""
    print("\n🔥 Installing PyTorch with GPU support...")
    
    if has_nvidia_gpu and platform.system() == "Windows":
        print("   Installing PyTorch with CUDA 12.1 support...")
        pytorch_install_cmd = [
            pip_cmd, "install", "torch", "torchvision", "torchaudio", 
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ]
    elif has_nvidia_gpu and platform.system() == "Linux":
        print("   Installing PyTorch with CUDA 12.1 support...")
        pytorch_install_cmd = [
            pip_cmd, "install", "torch", "torchvision", "torchaudio", 
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ]
    elif platform.system() == "Darwin":  # macOS
        print("   Installing PyTorch with MPS (Apple Silicon) support...")
        pytorch_install_cmd = [
            pip_cmd, "install", "torch", "torchvision", "torchaudio"
        ]
    else:
        print("   Installing PyTorch CPU-only version...")
        pytorch_install_cmd = [
            pip_cmd, "install", "torch", "torchvision", "torchaudio", 
            "--index-url", "https://download.pytorch.org/whl/cpu"
        ]
    
    subprocess.run(pytorch_install_cmd, check=True)
    print("✅ PyTorch installed")


def install_dependencies():
    """Install all required dependencies"""
    pip_cmd = get_pip_command()
    print("\n📦 Installing dependencies...")
    
    # Upgrade pip first
    print("   Upgrading pip...")
    #subprocess.run([pip_cmd, "install", "--upgrade", "pip"], check=True)
    
    # Check for GPU support
    has_nvidia_gpu = check_gpu_support()
    
    # Install PyTorch first with appropriate GPU support
    install_pytorch_with_cuda(pip_cmd, has_nvidia_gpu)
    
    # Install other requirements
    print("   Installing other requirements...")
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
echo ========================================
echo GPU and PyTorch Information:
echo ========================================
python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'Device Count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}'); [print(f'Device {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count()) if torch.cuda.is_available()]"
echo ========================================
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
echo ========================================
echo DETAILED DEBUG INFORMATION:
echo ========================================
echo System Information:
systeminfo | findstr /C:"System Model" /C:"Processor"
echo.
echo NVIDIA Information:
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits 2>nul || echo No NVIDIA GPU found
echo.
echo PyTorch Information:
python -c "import torch, sys; print(f'Python Version: {sys.version}'); print(f'PyTorch Version: {torch.__version__}'); print(f'PyTorch CUDA Compiled Version: {torch.version.cuda}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Runtime Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'Device Count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}'); [print(f'Device {i}: {torch.cuda.get_device_name(i)} - Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f}GB') for i in range(torch.cuda.device_count()) if torch.cuda.is_available()]; print(f'Current Device: {torch.cuda.current_device() if torch.cuda.is_available() else \"CPU\"}');"
echo ========================================
echo.
echo Starting application...
python app.py
pause
'''
        with open("run_debug.bat", "w") as f:
            f.write(debug_script)
        print("   Also created: run_debug.bat (for detailed troubleshooting)")
        
    else:
        script_name = "run.sh"
        script_content = '''#!/bin/bash
echo "Starting PyPotteryInk..."
source .venv/bin/activate
echo ""
echo "========================================"
echo "GPU and PyTorch Information:"
echo "========================================"
python -c "import torch; device_type = 'MPS' if torch.backends.mps.is_available() else 'CUDA' if torch.cuda.is_available() else 'CPU'; print(f'PyTorch Version: {torch.__version__}'); print(f'Device Type: {device_type}'); [print(f'CUDA Device {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None"
echo "========================================"
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
    """Detect available GPU with enhanced information"""
    python_cmd = os.path.join(".venv", "Scripts", "python") if platform.system() == "Windows" else os.path.join(".venv", "bin", "python")
    
    detect_script = '''
import torch
import sys

print("=== GPU Detection Results ===")
print(f"PyTorch Version: {torch.__version__}")
print(f"Python Version: {sys.version}")

if torch.cuda.is_available():
    print(f"✅ CUDA Available: Yes")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}")
        print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
    
    # Test tensor creation
    try:
        test_tensor = torch.randn(10, 10).cuda()
        print("✅ GPU tensor creation: Success")
    except Exception as e:
        print(f"❌ GPU tensor creation failed: {e}")
        
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print("✅ Apple Silicon GPU (MPS) detected")
    try:
        test_tensor = torch.randn(10, 10).to('mps')
        print("✅ MPS tensor creation: Success")
    except Exception as e:
        print(f"❌ MPS tensor creation failed: {e}")
else:
    print("⚠️  No GPU detected - will use CPU")
    print("If you have an NVIDIA GPU, make sure:")
    print("  1. NVIDIA drivers are installed")
    print("  2. CUDA toolkit is installed")
    print("  3. PyTorch was installed with CUDA support")
'''
    
    print("\n🔍 Running enhanced GPU detection...")
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
            print("  For troubleshooting, use: run_debug.bat")
        else:
            print(f"  Run: ./{run_script}")
        print("\nThe web interface will open automatically in your browser.")
        print("\nIf GPU is not detected, run the debug script for detailed information.")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Installation failed: {e}")
        print("\nFor GPU issues, try running run_debug.bat for more information.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()