"""
================================================================================
TASK 4: IMAGE SEGMENTATION & MORPHOLOGICAL PROCESSING
================================================================================

Purpose:
    - Segment enhanced images using thresholding techniques
    - Apply morphological operations (dilation, erosion, opening, closing)
    - Improve object boundaries and region clarity
    - Extract and refine segmented regions

Delivered:
    - Segmented images using global and Otsu's thresholding
    - Morphologically processed images (dilated, eroded, opened, closed)
    - Visual comparison of segmentation stages
================================================================================
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils

def apply_global_thresholding(image, threshold=127):
    """
    Apply global (fixed) thresholding
    
    Args:
        image (ndarray): Input grayscale image
        threshold (int): Threshold value (0-255)
        
    Returns:
        ndarray: Binary segmented image
    """
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    print(f"✓ Global thresholding applied (threshold={threshold})")
    return binary

def apply_otsu_thresholding(image):
    """
    Apply Otsu's automatic thresholding
    
    Args:
        image (ndarray): Input grayscale image
        
    Returns:
        tuple: (binary_image, threshold_value)
    """
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"✓ Otsu's thresholding applied (auto-determined threshold)")
    return binary

def apply_adaptive_thresholding(image, block_size=11, c=2):
    """
    Apply adaptive thresholding
    
    Args:
        image (ndarray): Input grayscale image
        block_size (int): Size of pixel neighborhood (odd number)
        c (int): Constant subtracted from mean
        
    Returns:
        ndarray: Binary segmented image
    """
    binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, block_size, c)
    print(f"✓ Adaptive thresholding applied (block_size={block_size}, c={c})")
    return binary

def apply_dilation(image, kernel_size=5, iterations=1):
    """
    Apply dilation morphological operation
    
    Args:
        image (ndarray): Input binary image
        kernel_size (int): Size of morphological kernel
        iterations (int): Number of iterations
        
    Returns:
        ndarray: Dilated image
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(image, kernel, iterations=iterations)
    print(f"✓ Dilation applied (kernel={kernel_size}x{kernel_size}, iterations={iterations})")
    return dilated

def apply_erosion(image, kernel_size=5, iterations=1):
    """
    Apply erosion morphological operation
    
    Args:
        image (ndarray): Input binary image
        kernel_size (int): Size of morphological kernel
        iterations (int): Number of iterations
        
    Returns:
        ndarray: Eroded image
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    eroded = cv2.erode(image, kernel, iterations=iterations)
    print(f"✓ Erosion applied (kernel={kernel_size}x{kernel_size}, iterations={iterations})")
    return eroded

def apply_opening(image, kernel_size=5):
    """
    Apply opening morphological operation (erosion followed by dilation)
    - Removes small objects and noise
    
    Args:
        image (ndarray): Input binary image
        kernel_size (int): Size of morphological kernel
        
    Returns:
        ndarray: Opened image
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    print(f"✓ Opening applied (kernel={kernel_size}x{kernel_size})")
    return opened

def apply_closing(image, kernel_size=5):
    """
    Apply closing morphological operation (dilation followed by erosion)
    - Fills small holes and gaps
    
    Args:
        image (ndarray): Input binary image
        kernel_size (int): Size of morphological kernel
        
    Returns:
        ndarray: Closed image
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    print(f"✓ Closing applied (kernel={kernel_size}x{kernel_size})")
    return closed

def apply_gradient(image, kernel_size=5):
    """
    Apply morphological gradient (dilation - erosion)
    - Extracts object outlines
    
    Args:
        image (ndarray): Input binary image
        kernel_size (int): Size of morphological kernel
        
    Returns:
        ndarray: Gradient image
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    print(f"✓ Morphological gradient applied (kernel={kernel_size}x{kernel_size})")
    return gradient

def segment_and_morph(enhanced_image, output_dir="outputs"):
    """
    Complete segmentation and morphological processing pipeline
    
    Args:
        enhanced_image (ndarray): Input enhanced image
        output_dir (str): Output directory
        
    Returns:
        dict: Dictionary containing all segmented images
    """
    print("\n--- TASK 4: IMAGE SEGMENTATION & MORPHOLOGICAL PROCESSING ---\n")
    
    print("Step 1: Applying segmentation techniques...")
    
    # Apply different thresholding methods
    global_threshold = apply_global_thresholding(enhanced_image, threshold=127)
    otsu_threshold = apply_otsu_thresholding(enhanced_image)
    adaptive_threshold = apply_adaptive_thresholding(enhanced_image, block_size=11, c=2)
    
    print("\nStep 2: Applying morphological operations to Otsu's result...")
    
    # Apply morphological operations on best result (Otsu's)
    dilated = apply_dilation(otsu_threshold, kernel_size=5, iterations=1)
    eroded = apply_erosion(otsu_threshold, kernel_size=5, iterations=1)
    opened = apply_opening(otsu_threshold, kernel_size=5)
    closed = apply_closing(otsu_threshold, kernel_size=5)
    gradient = apply_gradient(otsu_threshold, kernel_size=5)
    
    # Store all segmented images
    segmented_images = {
        'global': global_threshold,
        'otsu': otsu_threshold,
        'adaptive': adaptive_threshold,
        'dilated': dilated,
        'eroded': eroded,
        'opened': opened,
        'closed': closed,
        'gradient': gradient
    }
    
    # Visualize segmentation stages
    display_segmentation(otsu_threshold, dilated, eroded, opened, closed, gradient, output_dir)
    
    print("\nSegmentation and morphological processing completed!")
    
    return segmented_images

def display_segmentation(otsu, dilated, eroded, opened, closed, gradient, output_dir="outputs"):
    """
    Display segmentation and morphological processing stages
    
    Args:
        otsu (ndarray): Otsu's threshold result
        dilated (ndarray): Dilated result
        eroded (ndarray): Eroded result
        opened (ndarray): Opened result
        closed (ndarray): Closed result
        gradient (ndarray): Gradient result
        output_dir (str): Output directory
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('TASK 4: Image Segmentation & Morphological Processing', 
                 fontsize=14, fontweight='bold')
    
    images = [otsu, dilated, eroded, opened, closed, gradient]
    titles = ["Otsu's Threshold", "Dilated", "Eroded", 
              "Opened", "Closed", "Morphological Gradient"]
    
    for ax, img, title in zip(axes.flat, images, titles):
        ax.imshow(img, cmap='gray')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save figure
    output_path = utils.ensure_dir(output_dir)
    plt.savefig(f"{output_path}/task4_segmentation.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Segmentation visualization saved: task4_segmentation.png")
