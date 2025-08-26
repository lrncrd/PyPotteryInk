"""Process image with real-time status updates for Gradio"""

import os
import sys
import io
import time
import queue
import threading
from PIL import Image, ImageEnhance
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import gc
import warnings
warnings.filterwarnings("ignore")


class StatusCapture:
    """Capture all console output and forward to queue"""
    def __init__(self, original_stream, status_queue):
        self.original = original_stream
        self.queue = status_queue
        self.buffer = ""
        
    def write(self, text):
        # Write to original stream
        self.original.write(text)
        self.original.flush()
        
        # Process text
        if text:
            self.buffer += text
            # Check for complete lines or progress updates
            if '\n' in text or '\r' in text or '%' in text:
                lines = self.buffer.split('\n')
                for line in lines[:-1]:
                    if line.strip():
                        self.queue.put(line.strip())
                self.buffer = lines[-1]
                
                # For progress bars (contain \r), send immediately
                if '\r' in text and self.buffer.strip():
                    self.queue.put(self.buffer.strip())
                    self.buffer = ""
    
    def flush(self):
        self.original.flush()
        
    def __getattr__(self, name):
        return getattr(self.original, name)


def process_image_with_queue(
        input_image,
        model_path,
        output_dir,
        use_fp16,
        contrast_scale,
        patch_size,
        overlap,
        status_queue
):
    """Process image and send updates via queue"""
    
    def log(msg):
        """Send message to queue"""
        status_queue.put(msg)
        print(msg)
    
    # Capture all output
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StatusCapture(old_stdout, status_queue)
    sys.stderr = StatusCapture(old_stderr, status_queue)
    
    try:
        log("🚀 Initializing pix2pix_turbo processing...")
        log(f"📁 Model path: {model_path}")
        log("⚙️ Configuration:")
        log(f"  - FP16 mode: {use_fp16}")
        log(f"  - Patch size: {patch_size}px")
        log(f"  - Overlap: {overlap}px")
        log(f"  - Contrast scale: {contrast_scale}")
        log(f"📂 Output directory: {output_dir}")
        
        # Check device
        if torch.cuda.is_available():
            device = torch.device("cuda")
            log(f"🔧 Using device: cuda ({torch.cuda.get_device_name(0)})")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            log("🔧 Using device: mps (Metal Performance Shaders)")
        else:
            device = torch.device("cpu")
            log("🔧 Using device: cpu")
        
        # Import model (this is where download happens)
        log("")
        log("📦 Loading Pix2Pix_Turbo model...")
        from models import Pix2Pix_Turbo
        
        log(f"Initializing Pix2Pix_Turbo on device: {device}")
        model = Pix2Pix_Turbo(pretrained_path=model_path, device=device)
        model.set_eval()
        
        # FP16 handling
        if use_fp16 and torch.cuda.is_available():
            model.half()
            log("✅ FP16 mode enabled (CUDA)")
        elif use_fp16 and torch.backends.mps.is_available():
            log("ℹ️ FP16 not supported on MPS, using FP32")
        elif use_fp16:
            log("⚠️ FP16 requested but not supported on CPU, using FP32")
        
        log("✅ Model loaded successfully")
        log("")
        
        # Process image
        log("📷 Processing input image...")
        log(f"📐 Original size: {input_image.size}")
        
        # Adjust dimensions
        width = input_image.width - input_image.width % 4
        height = input_image.height - input_image.height % 4
        input_image = input_image.resize((width, height), Image.BICUBIC)
        log(f"🔧 Adjusted to: {width}×{height}")
        
        # Calculate patches
        from ink import calculate_patches, get_patch_coordinates, create_blend_mask
        total_patches, patches_per_row, num_rows = calculate_patches(width, height, patch_size, overlap)
        log(f"🧩 Total patches to process: {total_patches} ({patches_per_row}×{num_rows})")
        log("")
        
        # Initialize output
        output_image = Image.new('RGB', (width, height))
        original_size = (input_image.width, input_image.height)
        
        # Process patches
        log("🎨 Processing patches...")
        
        with torch.no_grad():
            for idx in range(total_patches):
                # Update progress
                progress = (idx + 1) / total_patches * 100
                bar_length = 30
                filled = int(bar_length * progress / 100)
                bar = '█' * filled + '░' * (bar_length - filled)
                status_queue.put(f"Processing patches: {progress:3.0f}%|{bar}| {idx+1}/{total_patches}")
                
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
                output_patch = model(c_t, "make it ready for publication")
                patch_pil = transforms.ToPILImage()(output_patch[0].cpu() * 0.5 + 0.5)
                
                # Blend and paste
                mask = create_blend_mask(x_end - x_start, y_end - y_start, row, col, overlap)
                output_image.paste(patch_pil, (x_start, y_start), mask)
                
                # Clean up
                del c_t, output_patch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        log("")
        log("✅ Patch processing completed!")
        
        # Clean up
        log("🧹 Cleaning up resources...")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Save result
        output_path = os.path.join(output_dir, "output.png")
        output_image = output_image.resize(original_size, Image.BICUBIC)
        output_image = output_image.convert('L')
        output_image = ImageEnhance.Contrast(output_image).enhance(1.5)
        output_image.save(output_path)
        
        log(f"💾 Saved to: {output_path}")
        log("✨ Processing complete!")
        
        return output_path
        
    except Exception as e:
        log(f"❌ Error: {str(e)}")
        raise
    finally:
        # Restore stdout/stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr