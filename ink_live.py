"""Live console output for PyPotteryInk - simpler approach"""

import os
import sys
import io
import time
import threading
from contextlib import redirect_stdout, redirect_stderr
from PIL import Image, ImageEnhance
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import gc
import warnings
warnings.filterwarnings("ignore")


def process_image_with_live_output(
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
        yield_callback=None
):
    """Process image with live status updates via yield_callback"""
    
    def log(message):
        """Send log message via callback"""
        if yield_callback:
            yield_callback(message)
        print(message)  # Also print to console
    
    try:
        log("🚀 Initializing pix2pix_turbo processing...")
        log(f"📁 Model path: {model_path}")
        log("⚙️ Configuration:")
        log(f"  - FP16 mode: {use_fp16}")
        log(f"  - Patch size: {patch_size}px")
        log(f"  - Overlap: {overlap}px")
        log(f"  - Contrast scale: {contrast_scale}")
        
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            log(f"📂 Output directory: {output_dir}")
        
        # Check device
        if torch.cuda.is_available():
            device = torch.device("cuda")
            log(f"🔧 Using device: cuda ({torch.cuda.get_device_name(0)})")
            log(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            log("🔧 Using device: mps (Metal Performance Shaders)")
        else:
            device = torch.device("cpu")
            log("🔧 Using device: cpu")
        
        # Import and initialize model
        log(f"\n📦 Loading Pix2Pix_Turbo model...")
        
        # Capture stdout/stderr during model initialization
        captured_lines = []
        
        class CaptureOutput:
            def __init__(self, original, callback):
                self.original = original
                self.callback = callback
                
            def write(self, text):
                self.original.write(text)
                self.original.flush()
                if text.strip():
                    self.callback(text.strip())
                    
            def flush(self):
                self.original.flush()
                
            def __getattr__(self, name):
                return getattr(self.original, name)
        
        def capture_callback(text):
            # Handle download progress
            if 'diffusion_pytorch_model.safetensors' in text or '%|' in text:
                log(text)
            elif text and not text.startswith('\r'):
                log(text)
        
        # Temporarily redirect stdout/stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = CaptureOutput(old_stdout, capture_callback)
        sys.stderr = CaptureOutput(old_stderr, capture_callback)
        
        try:
            from models import Pix2Pix_Turbo
            log(f"Initializing Pix2Pix_Turbo on device: {device}")
            model = Pix2Pix_Turbo(pretrained_path=model_path, device=device)
            model.set_eval()
        finally:
            # Restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        
        # FP16 handling
        if use_fp16 and torch.cuda.is_available():
            model.half()
            log("✅ FP16 mode enabled (CUDA)")
        elif use_fp16 and torch.backends.mps.is_available():
            log("ℹ️ FP16 not supported on MPS, using FP32")
        elif use_fp16:
            log("⚠️ FP16 requested but not supported on CPU, using FP32")
        
        log("✅ Model loaded successfully\n")
        
        # Load and process image
        if isinstance(input_image_path_or_pil, str):
            input_image = Image.open(input_image_path_or_pil).convert('RGB')
            log(f"📷 Loaded image from: {input_image_path_or_pil}")
        else:
            input_image = input_image_path_or_pil.convert('RGB')
            log("📷 Using provided PIL image")
        
        log(f"📐 Original size: {input_image.size}")
        
        # Handle scaling
        if upscale != 1:
            if upscale > 1:
                new_size = (int(input_image.width * upscale), int(input_image.height * upscale))
                input_image = input_image.resize(new_size, Image.LANCZOS)
                log(f"⬆️ Upscaled to: {new_size}")
            else:
                new_size = (int(input_image.width * upscale), int(input_image.height * upscale))
                input_image = input_image.resize(new_size, Image.BICUBIC)
                log(f"⬇️ Downscaled to: {new_size}")
        
        original_size = (input_image.width, input_image.height)
        
        # Adjust dimensions
        width = input_image.width - input_image.width % 4
        height = input_image.height - input_image.height % 4
        input_image = input_image.resize((width, height), Image.BICUBIC)
        log(f"🔧 Adjusted to: {width}×{height}")
        
        # Calculate patches
        from ink import calculate_patches, get_patch_coordinates, create_blend_mask
        total_patches, patches_per_row, num_rows = calculate_patches(width, height, patch_size, overlap)
        log(f"\n🧩 Total patches to process: {total_patches} ({patches_per_row}×{num_rows})")
        
        # Initialize output
        output_image = Image.new('RGB', (width, height))
        
        # Process patches
        log("\n🎨 Processing patches...")
        
        with torch.no_grad():
            for idx in range(total_patches):
                # Update progress
                progress = (idx + 1) / total_patches * 100
                bar_length = 30
                filled = int(bar_length * progress / 100)
                bar = '█' * filled + '░' * (bar_length - filled)
                log(f"Processing patches: {progress:3.0f}%|{bar}| {idx+1}/{total_patches}")
                
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
        
        log("\n✅ Patch processing completed!")
        
        # Clean up model
        log("🧹 Cleaning up resources...")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Save or return result
        if return_pil:
            log("✅ Returning PIL image")
            return output_image
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
            log(f"💾 Saved to: {output_path}")
            log("✨ Processing complete!")
            
            return output_path
            
    except Exception as e:
        log(f"❌ Error: {str(e)}")
        raise