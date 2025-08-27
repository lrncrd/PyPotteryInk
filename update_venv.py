#!/usr/bin/env python3
"""
Update script for PyPotteryInk - updates existing virtual environment
"""

import os
import sys
import subprocess
import platform


def get_pip_command():
    """Get the correct pip command based on OS"""
    if platform.system() == "Windows":
        return os.path.join(".venv", "Scripts", "pip")
    else:
        return os.path.join(".venv", "bin", "pip")


def update_dependencies():
    """Update all required dependencies"""
    pip_cmd = get_pip_command()
    
    if not os.path.exists(pip_cmd):
        print("❌ Virtual environment not found. Please run install.py first.")
        sys.exit(1)
    
    print("\n📦 Updating dependencies...")
    
    # Upgrade pip first
    print("   Upgrading pip...")
    subprocess.run([pip_cmd, "install", "--upgrade", "pip"], check=True)
    
    # Update requirements
    print("   Updating requirements...")
    subprocess.run([pip_cmd, "install", "--upgrade", "-r", "requirements.txt"], check=True)
    print("✅ Dependencies updated")


def check_models():
    """Check if models exist"""
    print("\n🤖 Checking models...")
    
    model_path = os.path.join("models", "model_10k.pkl")
    if os.path.exists(model_path):
        print("   ✓ Default model found")
    else:
        print("   ⚠️  Default model not found")
        print("   Run the following to download:")
        if platform.system() == "Windows":
            print("   .venv\\Scripts\\python -c \"from utils import download_test_data; download_test_data()\"")
        else:
            print("   .venv/bin/python -c \"from utils import download_test_data; download_test_data()\"")


def main():
    """Main update process"""
    print("=================================")
    print("PyPotteryInk Update")
    print("=================================")
    print(f"Platform: {platform.system()} {platform.machine()}")
    
    try:
        update_dependencies()
        check_models()
        
        print("\n" + "="*50)
        print("✅ Update completed successfully!")
        print("="*50)
        print("\nTo run PyPotteryInk:")
        if platform.system() == "Windows":
            print("  Double-click run.bat or run it from terminal")
        else:
            print("  Run: ./run.sh")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Update failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()