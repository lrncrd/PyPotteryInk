#!/usr/bin/env python3
"""
Pre-download all PyPotteryInk models to avoid waiting during processing
"""

import os
import sys
import time
import requests
from tqdm import tqdm

# Model configurations
MODELS = {
    "10k Model": {
        "url": "https://huggingface.co/lrncrd/PyPotteryInk/resolve/main/model_10k.pkl?download=true",
        "filename": "model_10k.pkl",
        "size_mb": 38.3
    },
    "6h-MCG Model": {
        "url": "https://huggingface.co/lrncrd/PyPotteryInk/resolve/main/6h-MCG.pkl?download=true", 
        "filename": "6h-MCG.pkl",
        "size_mb": 38.3
    },
    "6h-MC Model": {
        "url": "https://huggingface.co/lrncrd/PyPotteryInk/resolve/main/6h-MC.pkl?download=true",
        "filename": "6h-MC.pkl",
        "size_mb": 38.3
    },
    "4h-PAINT Model": {
        "url": "https://huggingface.co/lrncrd/PyPotteryInk/resolve/main/4h-PAINT.pkl?download=true",
        "filename": "4h-PAINT.pkl",
        "size_mb": 38.3
    }
}

def download_file(url, filename, chunk_size=8192*4):  # Larger chunk size
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

def main():
    print("🚀 PyPotteryInk Model Downloader")
    print("=" * 50)
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Check which models are already downloaded
    print("\n📊 Checking existing models...")
    to_download = []
    
    for name, config in MODELS.items():
        filepath = os.path.join("models", config["filename"])
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / 1024 / 1024
            print(f"✅ {name}: Already exists ({size_mb:.1f} MB)")
        else:
            print(f"❌ {name}: Not found")
            to_download.append((name, config))
    
    if not to_download:
        print("\n✨ All models are already downloaded!")
        
        # Now trigger the diffusers model download
        print("\n📦 Checking Diffusers models...")
        print("This will download the large diffusion models if needed (3.46GB)")
        print("This only happens once per model.\n")
        
        try:
            # Import to trigger any missing downloads
            from models import Pix2Pix_Turbo
            import torch
            
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            
            # Try loading the first model to trigger diffusers download
            model_path = os.path.join("models", "model_10k.pkl")
            print(f"Initializing model to check diffusers cache...")
            model = Pix2Pix_Turbo(pretrained_path=model_path, device=device)
            print("✅ Diffusers models are ready!")
            
        except Exception as e:
            print(f"⚠️  Note: {e}")
            print("The diffusers models will download when you first use the app.")
            
        return
    
    # Download missing models
    print(f"\n📥 Downloading {len(to_download)} models...")
    
    for name, config in to_download:
        print(f"\n🔄 Downloading {name}...")
        filepath = os.path.join("models", config["filename"])
        
        start_time = time.time()
        try:
            download_file(config["url"], filepath)
            elapsed = time.time() - start_time
            print(f"✅ Downloaded in {elapsed:.1f} seconds")
        except Exception as e:
            print(f"❌ Failed: {e}")
            if os.path.exists(filepath):
                os.remove(filepath)
    
    print("\n✨ Model download complete!")
    
    # Try to trigger diffusers download
    print("\n📦 Initializing diffusers models...")
    print("This will download the large diffusion models if needed (3.46GB)")
    print("Note: This download might be slow due to server limitations.\n")
    
    try:
        from models import Pix2Pix_Turbo
        import torch
        
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model_path = os.path.join("models", "model_10k.pkl")
        
        print("Loading model to trigger diffusers download...")
        model = Pix2Pix_Turbo(pretrained_path=model_path, device=device)
        print("✅ All models are ready to use!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted!")
        print("The download will resume automatically when you use the app.")
    except Exception as e:
        print(f"⚠️  {e}")
        print("The diffusers models will download when you first use the app.")

if __name__ == "__main__":
    main()