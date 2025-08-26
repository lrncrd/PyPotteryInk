import os
import sys
import io
import re
import time
import threading
import contextlib
from PIL import Image, ImageEnhance
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import gc
from tqdm import tqdm
# Delayed import to avoid circular dependencies
import warnings
warnings.filterwarnings("ignore")

class ConsoleCapture:
    """Enhanced console capture for tqdm and other outputs"""
    def __init__(self, callback=None):
        self.callback = callback
        self.buffer = []
        self._lock = threading.Lock()
        self._original_write = None
        self._tqdm_pattern = re.compile(r'(\d+%)\|.*?\|')
        
    def write(self, text):
        if not text:
            return
            
        with self._lock:
            # Handle different types of output
            if '\r' in text or self._tqdm_pattern.search(text):
                # This is likely a progress bar update
                cleaned = text.strip().replace('\r', '')
                if self.buffer and any(char in self.buffer[-1] for char in ['%', '|', '█']):
                    # Update the last line
                    self.buffer[-1] = cleaned
                else:
                    self.buffer.append(cleaned)
            elif '\n' in text:
                # Multi-line text
                lines = text.strip().split('\n')
                self.buffer.extend([line for line in lines if line])
            elif text.strip():
                # Regular text
                self.buffer.append(text.strip())
                
        if self.callback:
            # Send immediate update
            self.callback('\n'.join(self.buffer[-50:]))  # Keep last 50 lines
            
    def flush(self):
        pass
    
    def isatty(self):
        return False
        
    def get_output(self):
        with self._lock:
            return '\n'.join(self.buffer)

@contextlib.contextmanager
def capture_all_output(callback=None):
    """Capture stdout, stderr, and tqdm output"""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    # Create capture objects
    capture = ConsoleCapture(callback)
    
    # Monkey patch tqdm to use our output
    old_tqdm_init = tqdm.__init__
    
    def patched_tqdm_init(self, *args, **kwargs):
        kwargs['file'] = capture
        old_tqdm_init(self, *args, **kwargs)
    
    tqdm.__init__ = patched_tqdm_init
    
    # Redirect stdout/stderr
    sys.stdout = capture
    sys.stderr = capture
    
    try:
        yield capture
    finally:
        # Restore everything
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        tqdm.__init__ = old_tqdm_init

def download_model_with_progress(model_url, model_path, capture):
    """Download model with detailed progress capture"""
    import requests
    
    capture.write(f"📥 Downloading model to: {model_path}")
    capture.write(f"🌐 URL: {model_url}")
    
    try:
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        # Use tqdm for progress
        with open(model_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, 
                     desc="Downloading model") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        capture.write("✅ Model download completed!")
        return True
        
    except Exception as e:
        capture.write(f"❌ Download failed: {str(e)}")
        return False

def process_single_image_with_console(
        input_image_path_or_pil,
        model_path,
        prompt="make it ready for publication",
        output_dir='output',
        use_fp16=False,
        output_name=None,
        contrast_scale=1,
        return_pil=False,
        patch_size=512,
        overlap=64,
        upscale=1,
        status_callback=None
):
    """Process image with full console output capture"""
    
    full_output = []
    
    def update_callback(text):
        nonlocal full_output
        full_output = text.split('\n')
        if status_callback:
            # Send update immediately
            status_callback(text)
            # Small delay to ensure UI updates
            time.sleep(0.01)
    
    with capture_all_output(update_callback) as capture:
        # Initial setup
        print("🚀 Initializing pix2pix_turbo processing...")
        print(f"📁 Model path: {model_path}")
        print("⚙️ Configuration:")
        print(f"  - FP16 mode: {use_fp16}")
        print(f"  - Patch size: {patch_size}px")
        print(f"  - Overlap: {overlap}px")
        print(f"  - Contrast scale: {contrast_scale}")
        print(f"  - Prompt: {prompt}")
        
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            print(f"📁 Output directory: {output_dir}")
        
        # Check device availability
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"🔧 Using device: cuda ({torch.cuda.get_device_name(0)})")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("🔧 Using device: mps (Metal Performance Shaders)")
        else:
            device = torch.device("cpu")
            print("🔧 Using device: cpu")
        
        # Check if model exists, download if needed
        if not os.path.exists(model_path):
            print(f"⚠️ Model not found at {model_path}")
            # You would need to provide the URL for the model
            # This is just an example
            model_dir = os.path.dirname(model_path)
            os.makedirs(model_dir, exist_ok=True)
            
        # Initialize model
        print(f"Initializing Pix2Pix_Turbo on device: {device}")
        from models import Pix2Pix_Turbo
        model = Pix2Pix_Turbo(pretrained_path=model_path, device=device)
        model.set_eval()
        
        # Only use FP16 if CUDA is available
        if use_fp16 and torch.cuda.is_available():
            model.half()
            print("✅ FP16 mode enabled (CUDA)")
        elif use_fp16 and torch.backends.mps.is_available():
            print("ℹ️ FP16 not supported on MPS, using FP32")
        elif use_fp16:
            print("⚠️ FP16 requested but not supported on CPU, using FP32")
        
        # Process input image
        if isinstance(input_image_path_or_pil, str):
            input_image = Image.open(input_image_path_or_pil).convert('RGB')
            print(f"📷 Loaded image from: {input_image_path_or_pil}")
        else:
            input_image = input_image_path_or_pil.convert('RGB')
            print("📷 Using provided PIL image")
        
        # Image preprocessing
        print(f"📐 Original size: {input_image.size}")
        
        if upscale != 1:
            if upscale > 1:
                new_size = (int(input_image.width * upscale), int(input_image.height * upscale))
                input_image = input_image.resize(new_size, Image.LANCZOS)
                print(f"⬆️ Upscaled to: {new_size}")
            else:
                new_size = (int(input_image.width * upscale), int(input_image.height * upscale))
                input_image = input_image.resize(new_size, Image.BICUBIC)
                print(f"⬇️ Downscaled to: {new_size}")
        
        original_size = (input_image.width, input_image.height)
        
        # Adjust dimensions for processing
        width = input_image.width - input_image.width % 4
        height = input_image.height - input_image.height % 4
        input_image = input_image.resize((width, height), Image.BICUBIC)
        print(f"🔧 Adjusted to: {width}×{height}")
        
        # Calculate patches
        from ink import calculate_patches, get_patch_coordinates, create_blend_mask
        total_patches, patches_per_row, num_rows = calculate_patches(width, height, patch_size, overlap)
        print(f"🧩 Total patches to process: {total_patches} ({patches_per_row}×{num_rows})")
        
        # Initialize output
        output_image = Image.new('RGB', (width, height))
        
        # Process patches
        print("\n🎨 Starting patch processing...")
        
        with torch.no_grad():
            with tqdm(total=total_patches, desc="Processing patches", unit="patch") as pbar:
                for idx in range(total_patches):
                    # Get patch coordinates
                    x_start, y_start, x_end, y_end, row, col = get_patch_coordinates(
                        idx, patches_per_row, num_rows, width, height, patch_size, overlap)
                    
                    # Extract and process patch
                    patch = input_image.crop((x_start, y_start, x_end, y_end))
                    patch = ImageEnhance.Contrast(patch).enhance(contrast_scale)
                    
                    c_t = F.to_tensor(patch).unsqueeze(0).to(device)
                    if use_fp16 and torch.cuda.is_available():
                        c_t = c_t.half()
                    
                    # Run model
                    output_patch = model(c_t, prompt)
                    patch_pil = transforms.ToPILImage()(output_patch[0].cpu() * 0.5 + 0.5)
                    
                    # Get blending mask and paste
                    mask = create_blend_mask(x_end - x_start, y_end - y_start, row, col, overlap)
                    output_image.paste(patch_pil, (x_start, y_start), mask)
                    
                    # Update progress
                    pbar.update(1)
                    
                    # Clean up
                    del c_t, output_patch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        print("✅ Patch processing completed!")
        
        # Clean up
        print("🧹 Cleaning up resources...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Save or return result
        if return_pil:
            print("✅ Returning PIL image")
            return output_image, capture.get_output()
        else:
            # Determine output filename
            if isinstance(input_image_path_or_pil, str):
                bname = os.path.basename(input_image_path_or_pil)
            else:
                bname = output_name if output_name else "output.png"
            
            output_path = os.path.join(output_dir, bname)
            
            # Restore original size
            output_image = output_image.resize(original_size, Image.BICUBIC)
            
            # Convert and enhance
            output_image = output_image.convert('L')
            output_image = ImageEnhance.Contrast(output_image).enhance(1.5)
            
            # Save
            output_image.save(output_path)
            print(f"💾 Saved to: {output_path}")
            print(f"✨ Processing complete!")
            
            return output_path, capture.get_output()