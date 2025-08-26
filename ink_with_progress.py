import os
from PIL import Image, ImageEnhance
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import gc
from models import Pix2Pix_Turbo
import sys
from io import StringIO


### remove warnings
import warnings
warnings.filterwarnings("ignore")

class ProgressCapture:
    """Capture console output and progress updates"""
    def __init__(self, callback=None):
        self.callback = callback
        self.messages = []
        
    def write(self, message):
        if message.strip():
            self.messages.append(message.strip())
            if self.callback:
                self.callback(message.strip())
    
    def get_messages(self):
        return "\n".join(self.messages)

def process_single_image_with_progress(
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
        progress_callback=None
):
    """Process a single image with progress updates"""
    
    # Create progress capture
    progress = ProgressCapture(progress_callback)
    
    # Initial setup and logging
    progress.write("🚀 Initializing pix2pix_turbo processing...")
    progress.write(f"📁 Model path: {model_path}")
    progress.write("⚙️ Configuration:")
    progress.write(f"  - FP16 mode: {use_fp16}")
    progress.write(f"  - Patch size: {patch_size}px")
    progress.write(f"  - Overlap: {overlap}px")
    progress.write(f"  - Contrast scale: {contrast_scale}")
    progress.write(f"  - Prompt: {prompt}")

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        progress.write(f"📁 Output directory created: {output_dir}")

    # Check device availability - support CUDA, MPS, or CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    progress.write(f"🔧 Using device: {device}")

    # Capture stdout during model initialization
    old_stdout = sys.stdout
    stdout_capture = StringIO()
    
    try:
        # Redirect stdout to capture initialization messages
        sys.stdout = stdout_capture
        
        # Initialize model
        model = Pix2Pix_Turbo(pretrained_path=model_path, device=device)
        model.set_eval()
        
        # Only use FP16 if CUDA is available
        if use_fp16 and torch.cuda.is_available():
            model.half()
        elif use_fp16:
            progress.write("⚠️ FP16 requested but not supported on this device, using FP32 instead")
        
        # Get captured output and send to progress
        sys.stdout = old_stdout
        captured = stdout_capture.getvalue()
        for line in captured.split('\n'):
            if line.strip():
                progress.write(line.strip())
                
    except Exception as e:
        sys.stdout = old_stdout
        progress.write(f"❌ Error initializing model: {str(e)}")
        raise
    
    # Process input image
    if isinstance(input_image_path_or_pil, str):
        input_image = Image.open(input_image_path_or_pil).convert('RGB')
    else:
        input_image = input_image_path_or_pil.convert('RGB')

    # Upscale image
    if upscale >= 1:
        input_image = input_image.resize((input_image.width * upscale, input_image.height * upscale), Image.LANCZOS)
        progress.write(f"  - Upscaled size: {input_image.size}")
    else:
        new_width = round(input_image.width * upscale)
        new_height = round(input_image.height * upscale)
        input_image = input_image.resize((new_width, new_height), Image.BICUBIC)
        progress.write(f"  - Downscaled size: {input_image.size}")

    original_size = (input_image.width, input_image.height)

    # Get dimensions
    width = input_image.width - input_image.width % 4
    height = input_image.height - input_image.height % 4
    input_image = input_image.resize((width, height), Image.BICUBIC)

    # Initialize output
    output_image = Image.new('RGB', (width, height))

    # Calculate patches
    from ink import calculate_patches, get_patch_coordinates, create_blend_mask
    total_patches, patches_per_row, num_rows = calculate_patches(width, height, patch_size, overlap)
    progress.write(f"🧩 Processing {total_patches} patches ({patches_per_row}x{num_rows})")

    try:
        with torch.no_grad():
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
                progress_percent = (idx + 1) / total_patches * 100
                progress.write(f"Processing patches: {idx + 1}/{total_patches} ({progress_percent:.1f}%)")

                del c_t, output_patch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    except Exception as e:
        progress.write(f"❌ Error during processing: {str(e)}")
        raise
    finally:
        progress.write("🧹 Cleaning up resources...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if return_pil:
        progress.write("✅ Processing complete! Returning PIL image")
        return output_image, progress.get_messages()
    else:
        if isinstance(input_image_path_or_pil, str):
            bname = os.path.basename(input_image_path_or_pil)
        else:
            bname = output_name if output_name else "output.png"
        output_path = os.path.join(output_dir, bname)
        # Back to original size
        output_image = output_image.resize(original_size, Image.BICUBIC)
        progress.write("✅ Processing complete! Saving to: {output_path}")
        output_image = output_image.convert('L')
        output_image = ImageEnhance.Contrast(output_image).enhance(1.5)
        output_image.save(output_path)
        progress.write("💾 Image saved successfully")
        return output_path, progress.get_messages()