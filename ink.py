import os
from PIL import Image, ImageEnhance
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import gc
from tqdm import tqdm
from models import Pix2Pix_Turbo
import matplotlib.pyplot as plt
import random
import time
import datetime
from utils import visualize_patches, print_disclosure_reminder

### remove warnings
import warnings
warnings.filterwarnings("ignore")

def run_diagnostics(
    input_folder,
    model_path,
    prompt = "make it ready for publication",
    patch_size=512,
    overlap=64,
    num_sample_images=5,
    contrast_values=[0.5, 0.75, 1, 1.5, 2, 3],
    output_dir='diagnostics',
):
    """
    Run diagnostic visualizations before main processing.
    Creates a diagnostic folder with patch visualizations and contrast tests.
    
    Args:
        input_folder (str): Folder containing images to process
        model_path (str): Path to the model
        prompt: (str) Text prompt for the model. Default: "make it ready for publication"
        patch_size (int): Size of patches for processing. Default: 512
        overlap (int): Overlap between patches. Default: 64
        num_sample_images (int): Number of sample images to use for diagnostics (max 5). Default: 5
        contrast_values (list): List of contrast values to test. Default: [0.5, 0.75, 1, 1.5, 2, 3]
        output_dir (str): Output directory for diagnostic images. Default: 'diagnostics'
   
    Returns:
        None
    """
    print("\n🔍 Running pre-processing diagnostics...")
    
    # Create diagnostics directory
    #diag_dir = os.path.join(output_dir, 'diagnostics')
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of images
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend([f for f in os.listdir(input_folder) if f.lower().endswith(ext)])
    
    if not image_files:
        print("❌ No images found for diagnostics!")
        return False
    
    # Select sample images
    num_samples = min(num_sample_images, len(image_files), 5)
    sample_images = random.sample(image_files, num_samples)
    
    print(f"\n📊 Running diagnostics on {num_samples} sample images...")
    
    # Initialize model for diagnostics
    print("🔄 Loading model for diagnostics...")
    # Check device availability - support CUDA, MPS, or CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = Pix2Pix_Turbo(pretrained_path=model_path, device=device)
    model.set_eval()
    # Note: FP16 (half precision) is not well supported on MPS yet
    if torch.cuda.is_available():
        model.half()  # Using FP16 for diagnostics only on CUDA GPU
    
    for idx, img_file in enumerate(sample_images):
        print(f"\n🖼️ Analyzing sample image {idx+1}/{num_samples}: {img_file}")
        input_path = os.path.join(input_folder, img_file)
        
        try:
            # 1. Patch visualization
            print("  📐 Generating patch visualization...")
            patch_viz_path = os.path.join(output_dir, f'patches_{idx+1}.png')
            visualize_patches(input_path, patch_size, overlap, patch_viz_path)
            
            # 2. Contrast analysis
            print("  🎨 Analyzing contrast effects...")
            fig, ax = plt.subplots(len(contrast_values), 2, figsize=(10, 4*len(contrast_values)))
            
            for i, value in enumerate(contrast_values):
                # Load and process original image
                my_img = Image.open(input_path).convert('RGB')
                img_contrast = ImageEnhance.Contrast(my_img).enhance(value)
                original_img_size = my_img.size
                
                # Display contrast-enhanced input
                ax[i, 0].imshow(img_contrast)
                ax[i, 0].set_ylabel(f"Contrast: {value}")
                
                # Process with model
                print(f"    Processing contrast value {value}...")
                res_img = process_single_image(
                    input_image_path_or_pil=my_img,
                    prompt=prompt,
                    model_path=model_path,
                    patch_size=patch_size,
                    contrast_scale=value,
                    overlap=overlap,
                    use_fp16=True,
                    return_pil=True,
                    output_dir=None
                )
                
                # Display model output
                ax[i, 1].imshow(res_img.resize(original_img_size))
                
                # Remove axes
                ax[i, 0].set_xticks([])
                ax[i, 0].set_yticks([])
                ax[i, 1].set_xticks([])
                ax[i, 1].set_yticks([])
            
            # Add titles
            ax[0, 0].set_title("Input with Contrast")
            ax[0, 1].set_title("Model Output")
            
            # Save contrast analysis
            contrast_path = os.path.join(output_dir, f'contrast_analysis_{idx+1}.png')
            ###
            plt.tight_layout()
            ###
            plt.savefig(contrast_path, bbox_inches='tight', dpi=300)
            plt.close()
            
            # 3. Generate summary info
            print("  📝 Generating image summary...")
            img = Image.open(input_path)
            summary = {
                "filename": img_file,
                "original_size": img.size,
                "num_patches": ((img.width + patch_size - overlap - 1) // (patch_size - overlap)) * 
                              ((img.height + patch_size - overlap - 1) // (patch_size - overlap)),
                "estimated_memory": f"{(img.width * img.height * 4) / (1024*1024):.2f}MB"
            }
            
            # Save summary to text file
            with open(os.path.join(output_dir, f'summary_{idx+1}.txt'), 'w') as f:
                for key, value in summary.items():
                    f.write(f"{key}: {value}\n")
            
        except Exception as e:
            print(f"❌ Error during diagnostics for {img_file}: {str(e)}")
            continue
    
    print("\n✅ Diagnostics complete!")
    print(f"📁 Results saved in: {output_dir}")

   

def calculate_patches(width, height, patch_size=512, overlap=64):
    """
    Calculate patch coordinates and dimensions for image processing.
    
    Args:
        width (int): Image width
        height (int): Image height
        patch_size (int): Size of each patch. Default: 512
        overlap (int): Overlap between patches. Default: 64
        
    Returns:
        list: List of patch information dictionaries containing:
            - x_start: Starting x coordinate
            - y_start: Starting y coordinate
            - width: Patch width
            - height: Patch height
            - row: Row number
            - col: Column number
    """
    stride = patch_size - overlap
    
    # Calculate patches per row and number of rows
    patches_per_row = (width - overlap) // (patch_size - overlap)
    if width > patches_per_row * (patch_size - overlap) + overlap:
        patches_per_row += 1
    
    num_rows = (height - overlap) // (patch_size - overlap)
    if height > num_rows * (patch_size - overlap) + overlap:
        num_rows += 1
    
    patches = []
    
    for row in range(num_rows):
        for col in range(patches_per_row):
            # Calculate base coordinates
            x_start = col * stride
            y_start = row * stride
            
            # Handle last column
            if col == patches_per_row - 1:
                x_end = width
                x_start = max(0, x_end - patch_size)
            else:
                x_end = min(x_start + patch_size, width)
            
            # Handle last row
            if row == num_rows - 1:
                y_end = height
                y_start = max(0, y_end - patch_size)
            else:
                y_end = min(y_start + patch_size, height)
            
            patch_info = {
                'x_start': x_start,
                'y_start': y_start,
                'width': x_end - x_start,
                'height': y_end - y_start,
                'row': row,
                'col': col
            }
            patches.append(patch_info)
    
    return patches, patches_per_row, num_rows



def calculate_patches(width, height, patch_size=512, overlap=64):
    """
    Calculate patch coordinates and dimensions for image processing.
    
    Args:
        width (int): Image width
        height (int): Image height
        patch_size (int): Size of each patch. Default: 512
        overlap (int): Overlap between patches. Default: 64
        
    Returns:
        tuple: (total_patches, patches_per_row, num_rows)
    """
    stride = patch_size - overlap
    
    # Calculate patches per row and number of rows
    patches_per_row = (width - overlap) // (patch_size - overlap)
    if width > patches_per_row * (patch_size - overlap) + overlap:
        patches_per_row += 1
    
    num_rows = (height - overlap) // (patch_size - overlap)
    if height > num_rows * (patch_size - overlap) + overlap:
        num_rows += 1

    total_patches = patches_per_row * num_rows
    
    return total_patches, patches_per_row, num_rows

def get_patch_coordinates(idx, patches_per_row, num_rows, width, height, patch_size, overlap):
    """
    Calculate coordinates for a specific patch.
    
    Args:
        idx (int): Patch index
        patches_per_row (int): Number of patches per row
        num_rows (int): Number of rows
        width (int): Image width
        height (int): Image height
        patch_size (int): Size of each patch
        overlap (int): Overlap between patches
        
    Returns:
        tuple: (x_start, y_start, x_end, y_end, row, col)
    """
    stride = patch_size - overlap
    row = idx // patches_per_row
    col = idx % patches_per_row
    
    # Calculate base coordinates
    x_start = col * stride
    y_start = row * stride
    
    # Handle last column
    if col == patches_per_row - 1:
        x_end = width
        x_start = max(0, x_end - patch_size)
    else:
        x_end = min(x_start + patch_size, width)
    
    # Handle last row
    if row == num_rows - 1:
        y_end = height
        y_start = max(0, y_end - patch_size)
    else:
        y_end = min(y_start + patch_size, height)
    
    return x_start, y_start, x_end, y_end, row, col

def create_blend_mask(patch_width, patch_height, row, col, overlap):
    """
    Create a blending mask for patch seamless integration.
    
    Args:
        patch_width (int): Width of the patch
        patch_height (int): Height of the patch
        row (int): Row number of the patch
        col (int): Column number of the patch
        overlap (int): Overlap size between patches
        
    Returns:
        PIL.Image: Blending mask
    """
    if row == 0 and col == 0:
        return None
        
    mask = Image.new('L', (patch_width, patch_height), 255)
    for k in range(overlap):
        alpha = int(255 * k / overlap)
        if col > 0:  # Blend left edge
            mask.paste(alpha, (k, 0, k+1, patch_height))
        if row > 0:  # Blend top edge
            mask.paste(alpha, (0, k, patch_width, k+1))
    
    return mask


def process_single_image(
        input_image_path_or_pil,
        model_path,
        prompt = "make it ready for publication",
        output_dir='output',
        use_fp16=False,
        output_name=None,
        contrast_scale=1,
        return_pil=False,
        patch_size=512,
        overlap=64,
        upscale=1,
):
    """
    Process a single image with modified pix2pix_turbo using improved patch strategy.
    """
    # Initial setup and logging (unchanged)
    print_disclosure_reminder()
    print(f"\n🚀 Initializing pix2pix_turbo processing...")
    print(f"📁 Model path: {model_path}")
    print(f"⚙️ Configuration:")
    print(f"  - FP16 mode: {use_fp16}")
    print(f"  - Patch size: {patch_size}px")
    print(f"  - Overlap: {overlap}px")
    print(f"  - Contrast scale: {contrast_scale}")
    print(f"  - Prompt: {prompt}")

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Output directory created: {output_dir}")

    # Check device availability - support CUDA, MPS, or CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"🔧 Using device: {device}")

    # Initialize model with proper device handling
    model = Pix2Pix_Turbo(pretrained_path=model_path, device=device)
    model.set_eval()
    # Only use FP16 if CUDA is available
    if use_fp16 and torch.cuda.is_available():
        model.half()
    elif use_fp16:
        print("⚠️ FP16 requested but CUDA not available, using FP32 instead")

    # Process input image
    if isinstance(input_image_path_or_pil, str):
        input_image = Image.open(input_image_path_or_pil).convert('RGB')
    else:
        input_image = input_image_path_or_pil.convert('RGB')


    # Upscale image

    if upscale >= 1:
        input_image = input_image.resize((input_image.width * upscale, input_image.height * upscale), Image.LANCZOS)
        print(f"  - Upscaled size: {input_image.size}")
    else:
        new_width = round(input_image.width * upscale)
        new_height = round(input_image.height * upscale)

        # Resize con filtro LANCZOS (ideale per riduzione di dimensione)
        input_image = input_image.resize((new_width, new_height), Image.BICUBIC)
        print(f"  - Downscaled size: {input_image.size}")

    original_size = (input_image.width, input_image.height)

    # Get dimensions
    width = input_image.width - input_image.width % 4
    height = input_image.height - input_image.height % 4
    input_image = input_image.resize((width, height), Image.BICUBIC)

    # Initialize output
    output_image = Image.new('RGB', (width, height))

    # Calculate patches
    total_patches, patches_per_row, num_rows = calculate_patches(width, height, patch_size, overlap)

    try:
        with torch.no_grad():
            for idx in tqdm(range(total_patches), desc=f"Processing image"):
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

                del c_t, output_patch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    except Exception as e:
        print(f"\n❌ Error during processing: {str(e)}")
        raise
    finally:
        print("\n🧹 Cleaning up resources...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if return_pil:
        print(f"\n✅ Processing complete! Returning PIL image \n ---------------------------------")
        return output_image
    else:
        if isinstance(input_image_path_or_pil, str):
            bname = os.path.basename(input_image_path_or_pil)
        else:
            bname = output_name if output_name else "output.png"
        output_path = os.path.join(output_dir, bname)
                        # Back to original size
        output_image = output_image.resize(original_size, Image.BICUBIC)
        print(f"\n✅ Processing complete! Saving to: {output_path}")
        output_image = output_image.convert('L')
        output_image = ImageEnhance.Contrast(output_image).enhance(1.5)
        output_image.save(output_path)
        print(f"💾 Image saved successfully")
        return output_path
    

def process_folder(
    input_folder,
    model_path,
    prompt = "make it ready for publication",
    output_dir='output',
    use_fp16=False,
    contrast_scale=1,
    patch_size=512,
    overlap=64,
    file_extensions=('.jpg', '.jpeg', '.png'),
    upscale=1,
):
    """
    Process a folder of images using improved patch strategy
    """
    # Initial setup and logging
    print_disclosure_reminder()
    print(f"\n🚀 Initializing batch processing with pix2pix_turbo...")
    print(f"📁 Input folder: {input_folder}")
    print(f"📁 Model path: {model_path}")
    print(f"📁 Output directory: {output_dir}")
    print(f"\n⚙️ Configuration:")
    print(f"  - FP16 mode: {use_fp16}")
    print(f"  - Patch size: {patch_size}px")
    print(f"  - Overlap: {overlap}px")
    print(f"  - Contrast scale: {contrast_scale}")
    print(f"  - Upscale: {upscale}x")
    print(f"  - Prompt: {prompt}")

    # Setup directories
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Get list of images
    image_files = []
    for ext in file_extensions:
        image_files.extend([f for f in os.listdir(input_folder) if f.lower().endswith(ext)])
    
    if not image_files:
        print("❌ No images found in input folder!")
        return
    
    print(f"\n📸 Found {len(image_files)} images to process")

    # Initialize model
    print("\n🔄 Loading model...")
    # Check device availability - support CUDA, MPS, or CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = Pix2Pix_Turbo(pretrained_path=model_path, device=device)
    model.set_eval()
    # Note: FP16 is only supported on CUDA, not on MPS yet
    if use_fp16 and torch.cuda.is_available():
        print("🚀 Converting model to FP16")
        model.half()
    elif use_fp16:
        print("⚠️ FP16 requested but not supported on this device, using FP32 instead")
    print("✅ Model loaded successfully")

    # Process statistics
    successful_conversions = 0
    failed_conversions = 0
    failed_files = []
    processing_times = []

    # Create log file
    log_file = os.path.join(log_dir, f'processing_log_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    
    with open(log_file, 'w') as log:
        # Write initial configuration to log
        log.write(f"Processing started at: {datetime.datetime.now()}\n")
        log.write(f"Configuration:\n")
        log.write(f"- Input folder: {input_folder}\n")
        log.write(f"- Output directory: {output_dir}\n")
        log.write(f"- Model path: {model_path}\n")
        log.write(f"- FP16 mode: {use_fp16}\n")
        log.write(f"- Patch size: {patch_size}px\n")
        log.write(f"- Overlap: {overlap}px\n")
        log.write(f"- Contrast scale: {contrast_scale}\n")
        log.write(f"- Upscale: {upscale}x\n")
        log.write(f"- Prompt: {prompt}\n\n")

        # Process each image
        for idx, image_file in enumerate(image_files, 1):
            input_path = os.path.join(input_folder, image_file)
            print(f"\n\n🖼️ Processing image {idx}/{len(image_files)}: {image_file}")
            log.write(f"\nProcessing image {idx}/{len(image_files)}: {image_file}\n")
            
            start_time = time.time()
            
            try:
                # Load image
                print(f"📥 Loading image: {image_file}")
                input_image = Image.open(input_path).convert('RGB')


                original_size = (input_image.width, input_image.height)
                print(f"  - Original size: {original_size}")
                log.write(f"Original size: {original_size}\n")

                # Upscale or downscale image

                if upscale >= 1:
                    input_image = input_image.resize((input_image.width * upscale, input_image.height * upscale), Image.LANCZOS)
                    print(f"  - Upscaled size: {input_image.size}")
                else:
                    new_width = round(input_image.width * upscale)
                    new_height = round(input_image.height * upscale)

                    # Resize con filtro LANCZOS (ideale per riduzione di dimensione)
                    input_image = input_image.resize((new_width, new_height), Image.BICUBIC)
                    print(f"  - Downscaled size: {input_image.size}")

                # Calculate dimensions
                width = input_image.width - input_image.width % 4
                height = input_image.height - input_image.height % 4
                input_image = input_image.resize((width, height), Image.LANCZOS)
                print(f"  - Processing size: ({width}, {height})")
                log.write(f"Processing size: ({width}, {height})\n")

                # Initialize output
                output_image = Image.new('RGB', (width, height))
                
                # Calculate patches using utility function
                total_patches, patches_per_row, num_rows = calculate_patches(width, height, patch_size, overlap)
                print(f"🧩 Processing {total_patches} patches ({patches_per_row}x{num_rows})")
                log.write(f"Patches: {total_patches} ({patches_per_row}x{num_rows})\n")

                # Process patches
                with torch.no_grad():
                    for patch_idx in tqdm(range(total_patches), desc=f"Processing {image_file}"):
                        # Get patch coordinates using utility function
                        x_start, y_start, x_end, y_end, row, col = get_patch_coordinates(
                            patch_idx, patches_per_row, num_rows, width, height, patch_size, overlap
                        )

                        # Process patch
                        patch = input_image.crop((x_start, y_start, x_end, y_end))
                        patch = ImageEnhance.Contrast(patch).enhance(contrast_scale)
                        
                        c_t = F.to_tensor(patch).unsqueeze(0).to(device)
                        if use_fp16 and torch.cuda.is_available():
                            c_t = c_t.half()

                        # Run model
                        output_patch = model(c_t, prompt)
                        patch_pil = transforms.ToPILImage()(output_patch[0].cpu() * 0.5 + 0.5)

                        # Create blend mask using utility function and paste
                        mask = create_blend_mask(x_end - x_start, y_end - y_start, row, col, overlap)
                        output_image.paste(patch_pil, (x_start, y_start), mask)

                        del c_t, output_patch
                        torch.cuda.empty_cache()

                # Save output
                output_filename = os.path.join(output_dir, image_file)

                # Back to original size
                output_image = output_image.resize(original_size, Image.BICUBIC)

                output_image = output_image.convert('L')
                output_image = ImageEnhance.Contrast(output_image).enhance(1.5)
                output_image.save(output_filename)
                
                end_time = time.time()
                processing_time = end_time - start_time
                processing_times.append(processing_time)
                
                print(f"✅ Saved: {output_filename}")
                print(f"⏱️ Processing time: {processing_time:.2f}s")
                log.write(f"Processing time: {processing_time:.2f}s\n")
                log.write("Status: Success\n")
                
                successful_conversions += 1

            except Exception as e:
                print(f"❌ Error processing {image_file}: {str(e)}")
                log.write(f"Status: Failed - {str(e)}\n")
                failed_conversions += 1
                failed_files.append(image_file)
                continue

            finally:
                # Cleanup after each image
                gc.collect()
                torch.cuda.empty_cache()

        # Write final statistics to log
        log.write("\n\nFinal Statistics:\n")
        log.write(f"Successfully processed: {successful_conversions} images\n")
        log.write(f"Failed to process: {failed_conversions} images\n")
        if processing_times:
            log.write(f"Average processing time: {sum(processing_times)/len(processing_times):.2f}s\n")
        if failed_files:
            log.write("\nFailed files:\n")
            for file in failed_files:
                log.write(f"- {file}\n")

    # Print final statistics
    print("\n📊 Processing Summary:")
    print(f"  ✅ Successfully processed: {successful_conversions} images")
    print(f"  ❌ Failed to process: {failed_conversions} images")
    if processing_times:
        print(f"  ⏱️ Average processing time: {sum(processing_times)/len(processing_times):.2f}s")
    if failed_files:
        print("\n⚠️ Failed files:")
        for file in failed_files:
            print(f"  - {file}")
    
    print(f"\n📝 Detailed log saved to: {log_file}")
    print("\n🏁 Batch processing complete!")
    
    # Generate comparison visualizations
    print("\n📊 Generating comparison visualizations...")
    comparison_dir = os.path.join(output_dir, 'comparisons')
    os.makedirs(comparison_dir, exist_ok=True)

    # Create comparison plots
    for idx, image_file in enumerate(image_files, 1):
        if image_file not in failed_files:
            try:
                print(f"\rGenerating comparison {idx}/{len(image_files)}", end="")
                
                # Load original and processed images
                input_path = os.path.join(input_folder, image_file)
                output_path = os.path.join(output_dir, image_file)
                
                # Create comparison plot
                fig, ax = plt.subplots(1, 2, figsize=(15, 7))
                
                # Original image
                image_input = Image.open(input_path).convert('RGB')
                ax[0].imshow(image_input)
                ax[0].axis('off')
                ax[0].set_title('Original Image', pad=20)
                
                # Processed image
                image_output = Image.open(output_path)
                ax[1].imshow(image_output, cmap='gray')
                ax[1].axis('off')
                ax[1].set_title('Processed Image', pad=20)
                
                # Add main title
                plt.suptitle(f'Comparison for {image_file}\nSize: {image_input.size}', y=1.05)
                
                # Save comparison
                comparison_path = os.path.join(comparison_dir, f'comparison_{image_file}')
                plt.savefig(comparison_path, bbox_inches='tight', dpi=300, pad_inches=0.5)
                plt.close()

            except Exception as e:
                print(f"\n❌ Error generating comparison for {image_file}: {str(e)}")
                continue

    print(f"\n✅ Comparison visualizations saved in: {comparison_dir}")
    
    # Return results
    results = {
        'successful': successful_conversions,
        'failed': failed_conversions,
        'failed_files': failed_files,
        'average_time': sum(processing_times)/len(processing_times) if processing_times else 0,
        'log_file': log_file,
        'comparison_dir': comparison_dir
    }
    
    return results