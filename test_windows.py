"""
Automated test script for PyPotteryInk on Windows
Run this after installation to verify everything works correctly
"""
import os
import sys
import subprocess
import torch
from pathlib import Path
import time
import traceback


def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print('='*60)


def test_environment():
    """Test basic environment setup"""
    print_section("Environment Check")
    
    results = {
        "Python Version": sys.version.split()[0],
        "Python Executable": sys.executable,
        "Working Directory": os.getcwd(),
        "PyTorch Version": torch.__version__,
        "CUDA Available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        results["CUDA Device"] = torch.cuda.get_device_name(0)
        results["CUDA Memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        results["CUDA Version"] = torch.version.cuda
    
    # Check for CPU info
    try:
        import platform
        results["CPU"] = platform.processor()
        results["CPU Cores"] = os.cpu_count()
    except:
        pass
    
    # Display results
    for key, value in results.items():
        print(f"{key:.<30} {str(value):>28}")
    
    return results


def test_installation():
    """Test if all required components are installed"""
    print_section("Installation Check")
    
    checks = {
        "Virtual Environment": Path(".venv").exists(),
        "Run Script": Path("run.bat").exists(),
        "Models Directory": Path("models").exists(),
        "Default Model (10k)": Path("models/model_10k.pkl").exists(),
        "Test Image": Path("test_image.jpg").exists(),
        "App Script": Path("app.py").exists(),
        "Requirements File": Path("requirements.txt").exists(),
    }
    
    all_passed = True
    for item, exists in checks.items():
        status = "✓" if exists else "✗"
        print(f"{status} {item:.<50} {'Found' if exists else 'NOT FOUND':>7}")
        if not exists:
            all_passed = False
    
    return all_passed


def test_imports():
    """Test all required imports"""
    print_section("Import Check")
    
    modules = [
        ("gradio", "Web Interface"),
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("diffusers", "Diffusers"),
        ("PIL", "Pillow"),
        ("numpy", "NumPy"),
        ("matplotlib", "Matplotlib"),
        ("scipy", "SciPy"),
        ("tqdm", "Progress Bars"),
        ("huggingface_hub", "HuggingFace Hub"),
    ]
    
    # Test local modules
    local_modules = [
        ("ink", "Processing Module"),
        ("models", "Model Definitions"),
        ("preprocessing", "Preprocessing Tools"),
        ("postprocessing", "Postprocessing Tools"),
        ("app", "Gradio Application"),
    ]
    
    all_passed = True
    
    print("\nExternal Dependencies:")
    for module_name, description in modules:
        try:
            __import__(module_name)
            print(f"✓ {description:.<50} OK")
        except ImportError as e:
            print(f"✗ {description:.<50} ERROR: {e}")
            all_passed = False
    
    print("\nLocal Modules:")
    for module_name, description in local_modules:
        try:
            __import__(module_name)
            print(f"✓ {description:.<50} OK")
        except ImportError as e:
            print(f"✗ {description:.<50} ERROR: {e}")
            all_passed = False
    
    return all_passed


def test_gpu_operations():
    """Test GPU operations if available"""
    print_section("GPU Operations Test")
    
    if not torch.cuda.is_available():
        print("No CUDA GPU detected - skipping GPU tests")
        return True
    
    try:
        # Test basic GPU operations
        print("Testing GPU tensor operations...")
        
        # Create tensor on GPU
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        
        # Perform operation
        start = time.time()
        z = torch.matmul(x, y)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        print(f"✓ Matrix multiplication (1000x1000): {elapsed:.3f}s")
        
        # Test FP16 operations
        print("\nTesting FP16 operations...")
        x_fp16 = x.half()
        y_fp16 = y.half()
        
        start = time.time()
        z_fp16 = torch.matmul(x_fp16, y_fp16)
        torch.cuda.synchronize()
        elapsed_fp16 = time.time() - start
        
        print(f"✓ FP16 matrix multiplication: {elapsed_fp16:.3f}s")
        print(f"  Speed improvement: {elapsed/elapsed_fp16:.2f}x")
        
        # Memory info
        print(f"\nGPU Memory:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"  Reserved: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
        
        # Cleanup
        del x, y, z, x_fp16, y_fp16, z_fp16
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"✗ GPU test failed: {e}")
        traceback.print_exc()
        return False


def test_processing():
    """Test basic image processing functionality"""
    print_section("Image Processing Test")
    
    if not Path("test_image.jpg").exists():
        print("⚠ test_image.jpg not found - downloading...")
        try:
            from utils import download_test_data
            download_test_data()
        except:
            print("✗ Could not download test image")
            return False
    
    try:
        print("Loading processing module...")
        from ink import process_single_image, device, use_fp16_default
        
        print(f"Processing device: {device}")
        print(f"FP16 mode: {use_fp16_default}")
        
        # Create test output directory
        test_dir = Path("test_output")
        test_dir.mkdir(exist_ok=True)
        
        print("\nProcessing test image...")
        start_time = time.time()
        
        result = process_single_image(
            "test_image.jpg",
            "models/model_10k.pkl",
            output_dir=str(test_dir),
            return_pil=False,
            use_fp16=use_fp16_default
        )
        
        elapsed = time.time() - start_time
        
        # Check output
        output_path = Path(result)
        if output_path.exists():
            size_kb = output_path.stat().st_size / 1024
            print(f"✓ Processing successful!")
            print(f"  Time: {elapsed:.1f}s")
            print(f"  Output: {output_path}")
            print(f"  Size: {size_kb:.1f} KB")
            return True
        else:
            print("✗ Output file not created")
            return False
            
    except Exception as e:
        print(f"✗ Processing test failed: {e}")
        traceback.print_exc()
        return False


def test_gradio_app():
    """Test if Gradio app can be imported and initialized"""
    print_section("Gradio App Test")
    
    try:
        print("Importing Gradio app...")
        from app import demo
        
        print("✓ Gradio app imported successfully")
        print(f"  App title: {demo.title}")
        print(f"  Number of tabs: {len(demo.children)}")
        
        # Test theme
        print("\nChecking app configuration...")
        if hasattr(demo, 'theme'):
            print(f"  Theme: {demo.theme.name if hasattr(demo.theme, 'name') else 'Custom'}")
        
        return True
        
    except Exception as e:
        print(f"✗ Gradio app test failed: {e}")
        traceback.print_exc()
        return False


def generate_report(results):
    """Generate a test report"""
    print_section("Test Report Summary")
    
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
PyPotteryInk Windows Test Report
Generated: {timestamp}

System Information:
- Python: {results['env']['Python Version']}
- PyTorch: {results['env']['PyTorch Version']}
- CUDA: {'Yes - ' + results['env'].get('CUDA Device', 'N/A') if results['env']['CUDA Available'] else 'No'}
- Working Directory: {results['env']['Working Directory']}

Test Results:
- Installation Check: {'PASSED' if results['installation'] else 'FAILED'}
- Import Check: {'PASSED' if results['imports'] else 'FAILED'}
- GPU Operations: {'PASSED' if results['gpu'] else 'N/A'}
- Processing Test: {'PASSED' if results['processing'] else 'FAILED'}
- Gradio App: {'PASSED' if results['gradio'] else 'FAILED'}

Overall Status: {'ALL TESTS PASSED' if all(results.values() if isinstance(results.values(), bool) else True) else 'SOME TESTS FAILED'}
"""
    
    # Save report
    report_path = f"test_report_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(report)
    print(f"\nReport saved to: {report_path}")


def main():
    """Run all tests"""
    print("="*60)
    print("PyPotteryInk Windows Test Suite".center(60))
    print("="*60)
    
    results = {}
    
    # Run tests
    results['env'] = test_environment()
    results['installation'] = test_installation()
    results['imports'] = test_imports()
    results['gpu'] = test_gpu_operations()
    results['processing'] = test_processing()
    results['gradio'] = test_gradio_app()
    
    # Generate report
    generate_report(results)
    
    # Exit code
    all_critical_passed = results['installation'] and results['imports']
    sys.exit(0 if all_critical_passed else 1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)