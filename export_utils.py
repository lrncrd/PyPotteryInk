"""
Advanced export utilities for PyPotteryInk
Supports exporting individual elements and SVG conversion
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
import subprocess
from typing import List, Tuple, Optional


def extract_pottery_elements(image_path: str, output_dir: str, min_area: int = 1000) -> List[str]:
    """
    Extract individual pottery elements from a processed image
    
    Args:
        image_path: Path to the processed pottery image
        output_dir: Directory to save extracted elements
        min_area: Minimum area for detected elements (filters out noise)
    
    Returns:
        List of paths to extracted element images
    """
    # Create output directory for elements
    elements_dir = os.path.join(output_dir, 'extracted_elements')
    os.makedirs(elements_dir, exist_ok=True)
    
    # Read and prepare image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return []
    
    # Apply threshold to get binary image
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours (pottery elements)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    element_paths = []
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    for idx, contour in enumerate(contours):
        # Filter by area to remove noise
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Add padding
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img.shape[1] - x, w + 2 * padding)
        h = min(img.shape[0] - y, h + 2 * padding)
        
        # Extract element
        element_img = img[y:y+h, x:x+w]
        
        # Create mask for the specific contour
        mask = np.zeros_like(img)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        mask_crop = mask[y:y+h, x:x+w]
        
        # Apply mask to get clean element
        element_clean = cv2.bitwise_and(element_img, element_img, mask=mask_crop)
        
        # Convert to RGBA for transparency
        element_rgba = cv2.cvtColor(element_clean, cv2.COLOR_GRAY2RGBA)
        # Make background transparent
        element_rgba[:, :, 3] = mask_crop
        
        # Save element
        element_path = os.path.join(elements_dir, f'{base_name}_element_{idx+1}.png')
        cv2.imwrite(element_path, element_rgba)
        element_paths.append(element_path)
        
        print(f"  Extracted element {idx+1}: {w}x{h} pixels")
    
    print(f"✅ Extracted {len(element_paths)} pottery elements")
    return element_paths


def convert_to_svg(image_path: str, output_path: Optional[str] = None, 
                   simplify: bool = True, threshold: int = 127) -> str:
    """
    Convert a pottery image to SVG format using potrace
    
    Args:
        image_path: Path to input image
        output_path: Path for output SVG (if None, uses same name with .svg extension)
        simplify: Whether to simplify the paths
        threshold: Threshold for binarization
    
    Returns:
        Path to the generated SVG file
    """
    try:
        # Prepare output path
        if output_path is None:
            output_path = os.path.splitext(image_path)[0] + '.svg'
        
        # First convert to BMP for potrace
        img = Image.open(image_path)
        
        # Convert to grayscale if needed
        if img.mode != 'L':
            img = img.convert('L')
        
        # Apply threshold
        img = img.point(lambda x: 0 if x < threshold else 255, '1')
        
        # Save as BMP
        bmp_path = os.path.splitext(image_path)[0] + '_temp.bmp'
        img.save(bmp_path)
        
        # Run potrace
        cmd = ['potrace', '-s', '-o', output_path]
        if simplify:
            cmd.extend(['-t', '5'])  # Simplify paths
        cmd.append(bmp_path)
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Clean up temp file
        if os.path.exists(bmp_path):
            os.remove(bmp_path)
        
        if result.returncode == 0:
            print(f"✅ SVG created: {output_path}")
            return output_path
        else:
            print(f"❌ Error creating SVG: {result.stderr}")
            return None
            
    except FileNotFoundError:
        print("❌ Potrace not found. Please install potrace:")
        print("   macOS: brew install potrace")
        print("   Linux: sudo apt-get install potrace")
        print("   Windows: Download from http://potrace.sourceforge.net/")
        return None
    except Exception as e:
        print(f"❌ Error converting to SVG: {str(e)}")
        return None


def create_enhanced_comparison(input_path: str, output_path: str, 
                               elements_paths: List[str], output_dir: str) -> str:
    """
    Create an enhanced comparison image with individual elements
    
    Args:
        input_path: Path to original image
        output_path: Path to processed image
        elements_paths: List of paths to extracted elements
        output_dir: Directory to save the comparison
    
    Returns:
        Path to the enhanced comparison image
    """
    try:
        # Load images
        original = Image.open(input_path).convert('RGB')
        processed = Image.open(output_path).convert('RGB')
        
        # Calculate layout
        n_elements = min(len(elements_paths), 6)  # Show max 6 elements
        if n_elements > 0:
            cols = 3
            rows = 2 + ((n_elements - 1) // cols + 1)
        else:
            cols = 2
            rows = 1
        
        # Create figure
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(cols * 5, rows * 5))
        
        # Add original and processed images
        ax1 = plt.subplot(rows, cols, 1)
        ax1.imshow(original)
        ax1.set_title('Original', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        ax2 = plt.subplot(rows, cols, 2)
        ax2.imshow(processed, cmap='gray')
        ax2.set_title('Processed', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        # Add extracted elements
        for idx, element_path in enumerate(elements_paths[:n_elements]):
            if os.path.exists(element_path):
                element = Image.open(element_path)
                ax = plt.subplot(rows, cols, cols + 1 + idx)
                ax.imshow(element)
                ax.set_title(f'Element {idx + 1}', fontsize=12)
                ax.axis('off')
        
        # Add main title
        plt.suptitle(f'PyPotteryInk Processing Results\n{os.path.basename(input_path)}', 
                     fontsize=16, fontweight='bold')
        
        # Save enhanced comparison
        enhanced_path = os.path.join(output_dir, 'enhanced_comparisons', 
                                    f'enhanced_{os.path.basename(input_path)}')
        os.makedirs(os.path.dirname(enhanced_path), exist_ok=True)
        
        plt.tight_layout()
        plt.savefig(enhanced_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()
        
        return enhanced_path
        
    except Exception as e:
        print(f"Error creating enhanced comparison: {str(e)}")
        return None


def export_all_formats(image_path: str, output_dir: str) -> dict:
    """
    Export image in multiple formats with all enhancements
    
    Args:
        image_path: Path to the processed image
        output_dir: Base output directory
    
    Returns:
        Dictionary with paths to all exported files
    """
    results = {
        'png': image_path,  # Original is already PNG
        'svg': None,
        'elements': [],
        'elements_svg': []
    }
    
    # Create SVG version
    svg_dir = os.path.join(output_dir, 'svg_exports')
    os.makedirs(svg_dir, exist_ok=True)
    
    svg_path = os.path.join(svg_dir, os.path.basename(image_path).replace('.jpg', '.svg').replace('.png', '.svg'))
    results['svg'] = convert_to_svg(image_path, svg_path)
    
    # Extract elements
    results['elements'] = extract_pottery_elements(image_path, output_dir)
    
    # Convert elements to SVG
    for element_path in results['elements']:
        element_svg = convert_to_svg(element_path)
        if element_svg:
            results['elements_svg'].append(element_svg)
    
    return results