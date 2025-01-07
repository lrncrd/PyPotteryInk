import numpy as np
import os
from tqdm import tqdm
from skimage.morphology import remove_small_objects, disk, binary_dilation
from PIL import Image



def binarize_image(image, threshold=127):
    """
    Binarize an image using a threshold.
    
    Args:
        image (PIL.Image or np.array): PIL Image or numpy array
        threshold (int): Threshold value (0-255). Default: 127
    Returns:
        Binary PIL Image
    """
    # Convert to PIL Image if numpy array
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Convert to grayscale if not already
    if image.mode != 'L':
        image = image.convert('L')
    
    # Convert to numpy for processing
    img_array = np.array(image)
    
    # Apply threshold
    binary = (img_array > threshold).astype(np.uint8) * 255
    
    return Image.fromarray(binary)

def remove_white_background(image, threshold=250):
    """
    Remove white background from image and make it transparent.
    
    Args:
        image (PIL.Image): PIL Image
        threshold (int): Threshold for what is considered "white" (0-255). Default: 250
    Returns:
        PIL Image with transparent background
    """
    # Convert to RGBA if not already
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    # Get image data
    data = np.array(image)
    
    # Create alpha channel mask
    # Consider a pixel "white" if all RGB values are above threshold
    rgb = data[:,:,:3]
    mask = np.all(rgb > threshold, axis=2)
    
    # Set alpha channel to 0 for white pixels
    data[:,:,3] = np.where(mask, 0, 255)
    
    return Image.fromarray(data)

def process_image_binarize(image_path, binarize_threshold=127, white_threshold=250, save_path=None):
    """
    Combined function to binarize and remove white background.
    
    Args:
        image_path (str): Path to input image
        binarize_threshold (int): Threshold for binarization. Default: 127
        white_threshold (int): Threshold for white background removal. Default: 250
        save_path (str): Optional path to save processed image. Default: None
    Returns:
        Processed PIL Image
    """
    # Load image
    image = Image.open(image_path)
    
    # First binarize
    binary = binarize_image(image, binarize_threshold)
    
    # Then remove white background
    processed = remove_white_background(binary, white_threshold)
    
    # Save if path provided
    if save_path:
        processed.save(save_path, 'PNG')  # Use PNG to preserve transparency
        
    return processed

def binarize_folder_images(input_folder, binarize_threshold=127, white_threshold=250):
    """
    Binarize all images in a folder and save as PNG.

    Args:
        input_folder (str): Path to folder containing images
        binarize_threshold (int): Threshold for binarization. Default: 127
        white_threshold (int): Threshold for white background removal. Default: 250
    """
    output_folder = input_folder + "_binarized"
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(input_folder) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for image_file in tqdm(image_files):
        try:
            input_path = os.path.join(input_folder, image_file)
            
            # Change the extension to .png
            base_name = os.path.splitext(image_file)[0]  # Get filename without extension
            output_path = os.path.join(output_folder, f"{base_name}.png")
            
            # Process image
            process_image_binarize(
                input_path,
                binarize_threshold=binarize_threshold,
                white_threshold=white_threshold,
                save_path=output_path
            )
            
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
            
    print("Processing complete!")

def enhance_stippling(img, min_size=80, connectivity=2):
    """
    Isolate and enhance stippling patterns in archaeological drawings.
    
    Args:
        img (PIL.Image): Input image
        min_size (int): Minimum size threshold for objects. Default: 80
        connectivity (int): Connectivity parameter for morphological operations. Default: 2
        
    Returns:
        Tuple[PIL.Image, PIL.Image]: (processed image, isolated stippling pattern)
    """
    # Convert to grayscale if not already
    img = img.convert("L")
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Create binary mask (True for dots, False for background)
    binary_mask = img_array < 128
    
    # Apply remove_small_objects with boolean mask
    processed_mask = remove_small_objects(
        binary_mask, 
        min_size=min_size,
        connectivity=connectivity
    )
    
    # Isolate stippling pattern
    stippling_pattern = binary_mask ^ processed_mask  # XOR to get removed dots
    
    # Convert masks to 8-bit format
    output_array = processed_mask.astype(np.uint8) * 255
    stippling_array = stippling_pattern.astype(np.uint8) * 255
    
    return Image.fromarray(output_array), Image.fromarray(stippling_array)

def modify_stippling(processed_img, stippling_pattern, operation='dilate', intensity=0.5, opacity=1.0):
    """
    Modify stippling patterns in archaeological drawings through morphological operations
    and intensity modulation.
    
    Args:
        processed_img (PIL.Image): Base image without stippling
        stippling_pattern (PIL.Image): Isolated stippling pattern
        operation (str): Type of modification ('dilate', 'fade', or 'both'). Default: 'dilate'
        intensity (float): Morphological modification intensity (0.0-1.0). Default: 0.5
        opacity (float): Stippling opacity factor (0.0-1.0). Default: 1.0
        
    Returns:
        PIL.Image: Modified archaeological drawing with adjusted stippling
    """
    # Convert inputs to numpy arrays
    base_array = np.array(processed_img)
    points_array = np.array(stippling_pattern)
    
    if operation in ['dilate', 'both']:
        # Morphological modification
        radius = max(1, int(intensity * 2))
        structure = disk(radius, dtype=bool)
        modified_points = binary_dilation(points_array, footprint=structure).astype(np.uint8) * 255
    elif operation == 'fade':
        modified_points = points_array
    
    # Apply opacity modulation
    if operation in ['fade', 'both']:
        modified_points = (modified_points * opacity).astype(np.uint8)
    
    # Combine and invert
    combined = np.array(base_array + modified_points)
    inverted = 255 - combined
    
    return Image.fromarray(inverted)

def control_stippling(input_folder, min_size=50, connectivity=2, operation='fade', intensity = 0.5, opacity=0.5):
    """
    Control stippling patterns in archaeological drawings through morphological operations
    and intensity modulation.

    Args:
        input_folder (str): Path to folder containing images
        min_size (int): Minimum size threshold for objects. Default: 50
        connectivity (int): Connectivity parameter for morphological operations. Default: 2
        operation (str): Type of modification ('dilate', 'fade', or 'both'). Default: 'fade'
        intensity (float): Morphological modification intensity (0.0-1.0). Default: 0.5
        opacity (float): Stippling opacity factor (0.0-1.0). Default

    """
    output_folder = input_folder + "_dotting_modified"
    os.makedirs(output_folder, exist_ok=True)

    image_files = [f for f in os.listdir(input_folder) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for image_file in tqdm(image_files):
        try:
            input_path = os.path.join(input_folder, image_file)
            
            # Change the extension to .png
            base_name = os.path.splitext(image_file)[0]  # Get filename without extension
            output_path = os.path.join(output_folder, f"{base_name}.png")

            # Load image
            img = Image.open(input_path)

            # Enhance stippling
            processed_img, stippling_pattern = enhance_stippling(img, min_size=min_size, connectivity=connectivity)

            # Modify stippling
            if operation == 'dilate':
                modified = modify_stippling(processed_img, stippling_pattern, operation='dilate', intensity=intensity)
            elif operation == 'fade':
                modified = modify_stippling(processed_img, stippling_pattern, operation='fade', opacity=opacity)
            elif operation == 'both':
                modified = modify_stippling(processed_img, stippling_pattern, operation='both', intensity=intensity, opacity=opacity)

            # Save output
            modified.save(output_path)
           

        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")