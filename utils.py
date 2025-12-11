import numpy as np
# Set matplotlib to use non-interactive backend for macOS compatibility
import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot
import matplotlib.pyplot as plt
from PIL import Image

def visualize_patches(image_path_or_pil, patch_size=512, overlap=64, save_path=None):
    """
    Creates a visualization of how an image is divided into patches with proper grid layout
    
    Args:
        image_path_or_pil (str or PIL.Image): Path to image or PIL Image object
        patch_size (int): Size of each patch. Default: 512
        overlap (int): Overlap between patches. Default: 64
        save_path (str): Path to save visualization. If None, visualization is shown instead. Default: None
    """
    # Load image
    if isinstance(image_path_or_pil, str):
        img = Image.open(image_path_or_pil).convert('RGB')
    else:
        img = image_path_or_pil.convert('RGB')
    
    width = img.width
    height = img.height
    
    # Convert PIL image to numpy array for matplotlib
    img_array = np.array(img)
    
    # Calculate grid parameters
    stride = patch_size - overlap
    
    # Calculate patches per row and number of rows
    patches_per_row = (width - overlap) // (patch_size - overlap)
    if width > patches_per_row * (patch_size - overlap) + overlap:
        patches_per_row += 1
    
    # Calculate remaining width for last patch if needed
    last_col_width = width - (patches_per_row - 1) * stride
    if last_col_width > patch_size:
        patches_per_row += 1
        last_col_width = width - (patches_per_row - 1) * stride
    
    # Calculate row heights
    num_rows = (height - overlap) // (patch_size - overlap)
    if height > num_rows * (patch_size - overlap) + overlap:
        num_rows += 1
    
    # Calculate height of last row
    last_row_height = height - (num_rows - 1) * stride
    if last_row_height > patch_size:
        num_rows += 1
        last_row_height = height - (num_rows - 1) * stride
    
    # Create figure
    dpi = 100
    figsize = (width/dpi, height/dpi)
    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()
    
    # Display the image
    plt.imshow(img_array)
    
    # Draw patches
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    patch_count = 0
    
    for row in range(num_rows):
        current_height = patch_size if row < num_rows - 1 else last_row_height
        y_start = row * stride
        
        for col in range(patches_per_row):
            patch_count += 1
            color = colors[(row + col) % len(colors)]
            
            # Calculate x position and width
            x_start = col * stride
            current_width = patch_size if col < patches_per_row - 1 else last_col_width
            
            # Draw rectangle
            rect = plt.Rectangle(
                (x_start, y_start),
                current_width,
                current_height,
                fill=False,
                color=color,
                linewidth=2,
                alpha=0.8
            )
            ax.add_patch(rect)
            
            # Add label
            plt.text(
                x_start + current_width/2,
                y_start + current_height/2,
                f'P{patch_count}\n({int(current_width)}x{int(current_height)})',
                horizontalalignment='center',
                verticalalignment='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1),
                color='black',
                fontsize=10
            )
    
    # Set axis properties
    plt.axis('off')
    
    # Add title
    plt.title(f'Patch Division Pattern\nImage: {width}x{height}px, Patch: {patch_size}px, Overlap: {overlap}px',
              pad=20)
    
    # Add info box
    info_text = (f'Image size: {width}x{height}px\n'
                f'Standard patch size: {patch_size}x{patch_size}px\n'
                f'Additional patches: {int(last_col_width)}x{patch_size}px (width), '
                f'{patch_size}x{int(last_row_height)}px (height)\n'
                f'Total patches: {patch_count}\n'
                f'Effective stride: {stride}px\n'
                f'Patches: {patches_per_row}x{num_rows}')
    
    plt.text(10, height-10, info_text,
             fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8),
             verticalalignment='bottom')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def print_disclosure_reminder():
    """
    Prints the AI disclosure reminder when PyPotteryInk is run.
    This function should be called at the beginning of the main execution.
    """
    version = "2.0.1"  # Replace with version variable from your package
    
    print("\n" + "=" * 80)
    print(" 📢 PYPOTTERYINK AI DISCLOSURE REMINDER ".center(80, "="))
    print("=" * 80)
    print(
        f"\nYou are using PyPotteryInk version {version}, a Generative AI tool for translating\n"
        "archaeological pottery drawings into publication-ready illustrations.\n\n"
        "DISCLOSURE REQUIREMENT:\n"
        "When publishing or presenting results that use PyPotteryInk, please include:\n"
        "  1. The version of PyPotteryInk used\n"
        "  2. The specific model used (e.g., '10k Model' or '6h-MCG Model')\n"
        "  3. The number of images processed\n\n"
        "Suggested citation format:\n"
        f"\"This research utilized PyPotteryInk (version {version}) for the AI-assisted\n"
        "translation of [number] pottery drawings. PyPotteryInk is a generative AI tool\n"
        "developed by Lorenzo Cardarelli (https://github.com/lrncrd/PyPotteryInk).\"\n"
    )
    print("=" * 80)
    print("\n")