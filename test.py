import os
import shutil
from PIL import Image
import numpy as np
import requests
import torch
from tqdm import tqdm
from preprocessing import DatasetAnalyzer
from ink import process_single_image, process_folder, run_diagnostics
from postprocessing import binarize_folder_images, control_stippling
from models import Pix2Pix_Turbo
import time

MODEL_URL = "https://huggingface.co/lrncrd/PyPotteryInk/resolve/main/model_10k.pkl?download=true"
MODEL_PATH = "models/10k.pth"

### check if cuda is available

def check_cuda():
    """Check if CUDA is available"""
    if not torch.cuda.is_available():
        print("❌ CUDA is not available.")
        print("🚨 You need CUDA to run this code.")
        print("🚨 Please verify if your GPU is CUDA compatible")
        ### break the code
        raise Exception("CUDA not available.")
    else:
        print("✅ CUDA is available.")


def download_model(url, save_path):
    """Download the model if it doesn't exist"""
    if os.path.exists(save_path):
        print(f"✅ Model already exists at {save_path}")
        return
    
    print(f"📥 Downloading model from {url}...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Send a GET request to the URL
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Get the total file size
    total_size = int(response.headers.get('content-length', 0))
    
    # Open output file and write chunks with progress bar
    with open(save_path, 'wb') as f, tqdm(
        desc="Downloading",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            pbar.update(size)
    
    print(f"✅ Model downloaded successfully to {save_path}")

def create_test_image(size=(512, 512)):
    """Create a test image for testing"""
    # Create a pencil-like drawing with some patterns
    img = Image.new('RGB', size, color='white')
    pixels = img.load()
    
    # Add some lines and patterns
    for i in range(size[0]):
        for j in range(size[1]):
            if (i + j) % 50 < 3:  # Create lines
                pixels[i,j] = (100, 100, 100)
            if (i * j) % 100 < 5:  # Create dots
                pixels[i,j] = (80, 80, 80)
    
    return img

def setup_test_environment():
    """Setup test directories and files"""
    # Create test directories
    test_dirs = ['test_input', 'test_output', 'test_diagnostics', "test_output_binarized", "test_output_dotting_modified"]
    for d in test_dirs:
        os.makedirs(d, exist_ok=True)
    
    # Create test images
    for i in range(3):
        img = create_test_image()
        img.save(f'test_input/test_image_{i}.png')
    
    return test_dirs

def cleanup_test_environment(test_dirs):
    """Clean up test directories"""
    for d in test_dirs:
        if os.path.exists(d):
            shutil.rmtree(d)

def test_pipeline():
    """Main test function to verify pipeline functionality"""
    try:
        print("\n🧪 Starting PyPotteryInk pipeline tests...\n")
        
        # Setup test environment
        print("📁 Setting up test environment...")
        test_dirs = setup_test_environment()
        
        # Download model if needed
        #try:
            #download_model(MODEL_URL, MODEL_PATH)
        #except Exception as e:
            #print(f"❌ Model download failed: {str(e)}")
            #raise
        
        # Test 1: Model Loading
        print("\n🔄 Test 1: Testing model loading...")
        try:
            model = Pix2Pix_Turbo(pretrained_path=MODEL_PATH)
            model.set_eval()
            print("✅ Model loading successful")
        except Exception as e:
            print(f"❌ Model loading failed: {str(e)}")
            raise
        
        # Test 2: Preprocessing
        print("\n🔍 Test 2: Testing preprocessing...")
        try:
            analyzer = DatasetAnalyzer()
            test_img = create_test_image()
            metrics = analyzer.analyze_image(test_img)
            assert isinstance(metrics, dict)
            assert 'mean' in metrics
            assert 'std' in metrics
            print("✅ Preprocessing successful")
        except Exception as e:
            print(f"❌ Preprocessing failed: {str(e)}")
            raise
        
        # Test 3: Single Image Processing
        print("\n🖼️ Test 3: Testing single image processing...")
        try:
            result = process_single_image(
                "test_input/test_image_0.png",
                model_path=MODEL_PATH,
                output_dir="test_output",
                use_fp16=True
            )
            assert os.path.exists(result)
            print("✅ Single image processing successful")
        except Exception as e:
            print(f"❌ Single image processing failed: {str(e)}")
            raise
        
        # Test 4: Batch Processing
        print("\n📚 Test 4: Testing batch processing...")
        try:
            results = process_folder(
                "test_input",
                model_path=MODEL_PATH,
                output_dir="test_output",
                use_fp16=True
            )
            assert isinstance(results, dict)
            print("✅ Batch processing successful")
        except Exception as e:
            print(f"❌ Batch processing failed: {str(e)}")
            raise
        
        # Test 5: Diagnostics
        print("\n🔍 Test 5: Testing diagnostics...")
        try:
            run_diagnostics(
                "test_input",
                model_path=MODEL_PATH,
                output_dir="test_diagnostics",
                num_sample_images=2
            )
            print("✅ Diagnostics successful")
        except Exception as e:
            print(f"❌ Diagnostics failed: {str(e)}")
            raise
        
        # Test 6: Post-processing
        print("\n🎨 Test 6: Testing post-processing...")
        try:
            binarize_folder_images("test_output")
            control_stippling("test_output")
            print("✅ Post-processing successful")
        except Exception as e:
            print(f"❌ Post-processing failed: {str(e)}")
            raise
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {str(e)}")
        raise
        
    finally:
        # Cleanup
        print("\n🧹 Cleaning up test environment...")
        cleanup_test_environment(test_dirs)
        print("✅ Cleanup completed")

if __name__ == "__main__":
        
    check_cuda()
    # Download model if needed
    try:
        download_model(MODEL_URL, MODEL_PATH)
    except Exception as e:
        print(f"❌ Model download failed: {str(e)}")
        raise
    starting_time = time.time()
    test_pipeline()
    ending_time = time.time()

    print(f"\n🕒 Total time taken: {ending_time - starting_time:.2f} seconds")