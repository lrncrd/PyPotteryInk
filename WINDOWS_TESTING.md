# Windows Testing Guide for PyPotteryInk

This guide covers comprehensive testing procedures for PyPotteryInk on Windows systems.

## Test Environment Requirements

### Windows Versions
- Windows 10 (version 1903 or later)
- Windows 11

### Hardware Requirements
- **CPU**: Intel i5 or AMD Ryzen 5 (minimum)
- **RAM**: 8GB minimum, 16GB recommended
- **GPU** (optional but recommended):
  - NVIDIA GTX 1060 6GB or better
  - RTX series for FP16 support
- **Storage**: 5GB free space

### Software Prerequisites
- Python 3.8+ (3.10 or 3.11 recommended)
- Git for Windows
- CUDA Toolkit 11.7+ (for NVIDIA GPU support)

## Installation Testing

### 1. Clean Installation Test
```bash
# Clone repository
git clone https://github.com/lrncrd/PyPotteryInk.git
cd PyPotteryInk

# Run installation
python install.py
```

Expected outcomes:
- Virtual environment created in `.venv` folder
- All dependencies installed without errors
- Models downloaded to `models/` directory
- `run.bat` script created
- GPU detection shows correct hardware

### 2. Installation Error Recovery
Test recovery from common installation issues:
- Network interruption during model download
- Missing Python dependencies
- Incorrect Python version
- Permission errors

## Functional Testing

### 1. Application Launch
```bash
# Method 1: Double-click run.bat
# Method 2: From terminal
run.bat
```

Expected behavior:
- Terminal window opens showing startup messages
- Browser automatically opens to http://127.0.0.1:7860
- If port 7860 is occupied, next available port is used
- All tabs load correctly in the interface

### 2. Hardware Check Tab
Verify:
- Correct GPU detection (CUDA if NVIDIA present)
- Memory reporting accuracy
- CPU cores detection
- Python version display

### 3. Model Diagnostics Tab
Test with provided sample image:
1. Select `test_image.jpg`
2. Choose different models
3. Vary contrast settings (0.5, 1.0, 2.0)
4. Check patch visualization

Expected results:
- Patch preview displays correctly
- Contrast variations show in preview
- No crashes during processing

### 4. Preprocessing Tab
Test statistics calculation:
1. Select folder with multiple images
2. Enable "Generate visualization plots"
3. Run statistics calculation

Expected results:
- Summary table displays correctly
- All plots render (histograms, box plots, KDE)
- Statistics saved to specified location
- No matplotlib backend issues

### 5. Batch Processing Tab
Test full processing workflow:
1. Select input folder with test images
2. Choose model (start with 10k model)
3. Keep default settings
4. Process images

Monitor:
- Real-time progress updates in interface
- Console output for detailed logs
- Memory usage (Task Manager)
- GPU utilization (if available)

Expected results:
- All images processed successfully
- Output images saved to specified directory
- Comparison visualizations generated
- Processing log created

## Performance Testing

### 1. Benchmark Tests
Process `test_image.jpg` with different configurations:

| Configuration | Expected Time | Notes |
|--------------|---------------|-------|
| CUDA + FP16 | 50-60s | RTX cards only |
| CUDA + FP32 | 60-70s | All NVIDIA cards |
| CPU only | 300-400s | Fallback mode |

### 2. Memory Tests
Monitor memory usage during:
- Single image processing
- Batch processing (10+ images)
- Large image processing (>4000x4000 pixels)

### 3. Stability Tests
- Process 50+ images continuously
- Test with various image formats (JPG, PNG)
- Test with different image sizes
- Interrupt and resume processing

## Edge Case Testing

### 1. File System Tests
- Long file paths (>200 characters)
- Unicode characters in paths
- Network drives
- Read-only directories

### 2. Error Handling
- Corrupted image files
- Insufficient disk space
- Missing model files
- Invalid settings

### 3. Concurrent Usage
- Multiple instances running
- Shared model directory
- Port conflicts

## GPU-Specific Testing

### 1. NVIDIA CUDA Testing
```python
# Test CUDA availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
```

### 2. FP16 Testing
- Enable FP16 checkbox in interface
- Compare processing time with FP32
- Verify output quality

### 3. Multi-GPU Testing (if available)
- Verify only first GPU is used
- Check memory allocation

## Logging and Debugging

### 1. Check Log Files
```bash
# Processing logs location
output\logs\processing_log_*.txt
```

### 2. Enable Debug Mode
Set environment variable:
```bash
set PYTHONUNBUFFERED=1
python app.py
```

### 3. Common Windows Issues

**Issue**: "DLL load failed" errors
```bash
# Solution: Install Visual C++ Redistributables
# Download from Microsoft website
```

**Issue**: CUDA not detected
```bash
# Check CUDA installation
nvcc --version
nvidia-smi
```

**Issue**: Permission denied errors
```bash
# Run as administrator or check folder permissions
```

## Automated Test Script

Create `test_windows.py`:
```python
import os
import sys
import subprocess
import torch
from pathlib import Path

def test_environment():
    """Test basic environment setup"""
    print("Testing environment...")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Working directory: {os.getcwd()}")

def test_installation():
    """Test if all required components are installed"""
    print("\nTesting installation...")
    
    # Check virtual environment
    assert Path(".venv").exists(), "Virtual environment not found"
    
    # Check models
    assert Path("models/model_10k.pkl").exists(), "Default model not found"
    
    # Check run script
    assert Path("run.bat").exists(), "Run script not found"
    
    print("✓ All components installed")

def test_imports():
    """Test all required imports"""
    print("\nTesting imports...")
    try:
        import gradio
        import diffusers
        import PIL
        import numpy
        import matplotlib
        import scipy
        from ink import process_single_image
        from app import demo
        print("✓ All imports successful")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    return True

def test_processing():
    """Test basic image processing"""
    print("\nTesting image processing...")
    if not Path("test_image.jpg").exists():
        print("⚠ test_image.jpg not found, skipping processing test")
        return
    
    from ink import process_single_image
    try:
        result = process_single_image(
            "test_image.jpg",
            "models/model_10k.pkl",
            output_dir="test_output",
            return_pil=True
        )
        print("✓ Processing test successful")
    except Exception as e:
        print(f"✗ Processing error: {e}")

if __name__ == "__main__":
    print("=== PyPotteryInk Windows Test Suite ===\n")
    test_environment()
    test_installation()
    if test_imports():
        test_processing()
    print("\n=== Test Complete ===")
```

## Test Report Template

After testing, document results:

```markdown
## Windows Test Report

**Date**: [DATE]
**Tester**: [NAME]
**System**: Windows [VERSION]
**Hardware**: [CPU/GPU/RAM]

### Installation Test
- [ ] Clean installation successful
- [ ] Dependencies installed correctly
- [ ] Models downloaded
- [ ] GPU detected properly

### Functionality Test
- [ ] Application launches
- [ ] All tabs functional
- [ ] Image processing works
- [ ] Batch processing works
- [ ] Progress tracking displays

### Performance
- Single image time: [TIME]
- Memory usage: [RAM/VRAM]
- GPU utilization: [%]

### Issues Found
1. [Issue description and solution]

### Notes
[Additional observations]
```

## Continuous Testing

For ongoing development:
1. Run tests after each code change
2. Test on multiple Windows versions
3. Test with different GPU models
4. Monitor user-reported issues