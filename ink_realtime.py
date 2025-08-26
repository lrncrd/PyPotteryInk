"""Real-time console output capture for PyPotteryInk"""

import os
import sys
import io
import re
import time
import threading
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from PIL import Image, ImageEnhance
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import gc
from tqdm import tqdm
# Delayed import to avoid circular dependencies
import warnings
warnings.filterwarnings("ignore")


class RealTimeCapture(io.StringIO):
    """Capture output and send updates in real-time"""
    def __init__(self, callback=None, original_stream=None):
        super().__init__()
        self.callback = callback
        self.original_stream = original_stream
        self.buffer_lock = threading.Lock()
        self.lines = []
        
    def write(self, text):
        if not text:
            return
            
        # Write to original stream if available (so it shows in terminal)
        if self.original_stream and self.original_stream != self:
            self.original_stream.write(text)
            self.original_stream.flush()
        
        with self.buffer_lock:
            # Handle progress bars and carriage returns
            if '\r' in text:
                # Progress bar update - replace last line
                cleaned = text.replace('\r', '').strip()
                if cleaned and self.lines and any(char in self.lines[-1] for char in ['%', '|', '█', '░']):
                    self.lines[-1] = cleaned
                elif cleaned:
                    self.lines.append(cleaned)
            else:
                # Normal text - add new lines
                for line in text.split('\n'):
                    if line.strip():
                        self.lines.append(line.strip())
            
            # Keep only last 100 lines
            if len(self.lines) > 100:
                self.lines = self.lines[-100:]
        
        # Send update immediately (avoid recursion by temporarily disabling)
        if self.callback and not getattr(self, '_in_callback', False):
            self._in_callback = True
            try:
                self.callback('\n'.join(self.lines))
            finally:
                self._in_callback = False
            
    def flush(self):
        pass
        
    def get_output(self):
        with self.buffer_lock:
            return '\n'.join(self.lines)


def process_single_image_realtime(
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
    """Process image with real-time console output"""
    
    # Set up real-time capture
    capture_stdout = RealTimeCapture(callback=status_callback, original_stream=sys.stdout)
    capture_stderr = RealTimeCapture(callback=status_callback, original_stream=sys.stderr)
    
    # Start capturing
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = capture_stdout
    sys.stderr = capture_stderr
    
    try:
        # All output will now be captured and sent to callback in real-time
        print("🚀 Initializing pix2pix_turbo processing...")
        print(f"📁 Model path: {model_path}")
        print("⚙️ Configuration:")
        print(f"  - FP16 mode: {use_fp16}")
        print(f"  - Patch size: {patch_size}px")
        print(f"  - Overlap: {overlap}px")
        print(f"  - Contrast scale: {contrast_scale}")
        
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            print(f"📂 Output directory: {output_dir}")
        
        # Check device
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"🔧 Using device: cuda ({torch.cuda.get_device_name(0)})")
            print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("🔧 Using device: mps (Metal Performance Shaders)")
        else:
            device = torch.device("cpu")
            print("🔧 Using device: cpu")
        
        # Import and initialize model
        print(f"\n📦 Loading Pix2Pix_Turbo model...")
        from models import Pix2Pix_Turbo
        
        print(f"Initializing Pix2Pix_Turbo on device: {device}")
        model = Pix2Pix_Turbo(pretrained_path=model_path, device=device)
        model.set_eval()
        
        # FP16 handling
        if use_fp16 and torch.cuda.is_available():
            model.half()
            print("✅ FP16 mode enabled (CUDA)")
        elif use_fp16 and torch.backends.mps.is_available():
            print("ℹ️ FP16 not supported on MPS, using FP32")
        elif use_fp16:
            print("⚠️ FP16 requested but not supported on CPU, using FP32")
        
        print("✅ Model loaded successfully\n")
        
        # Load and process image
        if isinstance(input_image_path_or_pil, str):
            input_image = Image.open(input_image_path_or_pil).convert('RGB')
            print(f"📷 Loaded image from: {input_image_path_or_pil}")
        else:
            input_image = input_image_path_or_pil.convert('RGB')
            print("📷 Using provided PIL image")
        
        print(f"📐 Original size: {input_image.size}")
        
        # Handle scaling
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
        
        # Adjust dimensions
        width = input_image.width - input_image.width % 4
        height = input_image.height - input_image.height % 4
        input_image = input_image.resize((width, height), Image.BICUBIC)
        print(f"🔧 Adjusted to: {width}×{height}")
        
        # Calculate patches
        from ink import calculate_patches, get_patch_coordinates, create_blend_mask
        total_patches, patches_per_row, num_rows = calculate_patches(width, height, patch_size, overlap)
        print(f"\n🧩 Total patches to process: {total_patches} ({patches_per_row}×{num_rows})")
        
        # Initialize output
        output_image = Image.new('RGB', (width, height))
        
        # Process patches
        print("\n🎨 Processing patches...")
        
        with torch.no_grad():
            for idx in tqdm(range(total_patches), desc="Processing patches", unit="patch"):
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
                
                # Blend and paste
                mask = create_blend_mask(x_end - x_start, y_end - y_start, row, col, overlap)
                output_image.paste(patch_pil, (x_start, y_start), mask)
                
                # Clean up
                del c_t, output_patch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        print("\n✅ Patch processing completed!")
        
        # Clean up model
        print("🧹 Cleaning up resources...")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Save or return result
        if return_pil:
            print("✅ Returning PIL image")
            return output_image, capture_stdout.get_output()
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
            print("✨ Processing complete!")
            
            return output_path, capture_stdout.get_output()
            
    finally:
        # Restore stdout/stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr