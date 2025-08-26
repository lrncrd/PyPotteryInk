#!/usr/bin/env python3
"""Check Hugging Face cache location and size"""

import os
from pathlib import Path

def get_cache_dir():
    """Get Hugging Face cache directory"""
    # Check environment variable first
    cache_dir = os.getenv('HF_HOME', os.getenv('HUGGINGFACE_HUB_CACHE'))
    
    if not cache_dir:
        # Default locations
        home = Path.home()
        if os.name == 'nt':  # Windows
            cache_dir = home / '.cache' / 'huggingface'
        else:  # Mac/Linux
            cache_dir = home / '.cache' / 'huggingface'
    
    return Path(cache_dir)

def get_size(path):
    """Get total size of directory"""
    total = 0
    for entry in path.rglob('*'):
        if entry.is_file():
            total += entry.stat().st_size
    return total

def format_size(size):
    """Format size in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"

def main():
    print("🔍 Checking Hugging Face cache...")
    
    cache_dir = get_cache_dir()
    hub_dir = cache_dir / 'hub'
    
    print(f"\n📁 Cache location: {cache_dir}")
    
    if hub_dir.exists():
        print(f"📦 Hub cache exists at: {hub_dir}")
        
        # List all model directories
        print("\n📊 Cached models:")
        for model_dir in hub_dir.iterdir():
            if model_dir.is_dir():
                size = get_size(model_dir)
                if size > 0:
                    print(f"  - {model_dir.name}: {format_size(size)}")
        
        total_size = get_size(hub_dir)
        print(f"\n💾 Total cache size: {format_size(total_size)}")
        
        # Look for the large diffusers model
        print("\n🔍 Looking for img2img-turbo model...")
        for model_dir in hub_dir.iterdir():
            if 'img2img' in model_dir.name.lower() or 'gaparmar' in model_dir.name.lower():
                print(f"  ✅ Found: {model_dir.name}")
                # Check for the large safetensors file
                for file in model_dir.rglob('*.safetensors'):
                    if file.stat().st_size > 1e9:  # > 1GB
                        print(f"     - {file.name}: {format_size(file.stat().st_size)}")
    else:
        print("❌ No cache found yet. Models will download on first use.")
        
    print("\n💡 Tip: The large model (3.46GB) is downloaded to this location automatically.")
    print("   Once downloaded, it won't need to be downloaded again.")

if __name__ == "__main__":
    main()