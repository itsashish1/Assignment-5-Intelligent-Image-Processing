"""
================================================================================
TASK 2: IMAGE ACQUISITION & PREPROCESSING
================================================================================

Purpose:
    - Load images from disk or capture from webcam
    - Resize image to standard resolution (512×512)
    - Convert color images to grayscale
    - Display original and preprocessed images for comparison

Delivered:
    - Loaded and preprocessed images
    - Resized to standard dimensions
    - Grayscale conversion
    - Visual comparison display
================================================================================
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils

def load_image(image_path):
    """
    Load image from specified path
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        ndarray: Loaded image in BGR format
    """
    image = cv2.imread(image_path)
    
    if image is None:
        raise FileNotFoundError(f"Image not found at: {image_path}")
    
    print(f"✓ Image loaded: {image_path}")
    print(f"  Original dimensions: {image.shape}")
    
    return image

def resize_image(image, target_size=(512, 512)):
    """
    Resize image to standard resolution
    
    Args:
        image (ndarray): Input image
        target_size (tuple): Target dimensions (height, width)
        
    Returns:
        ndarray: Resized image
    """
    resized = cv2.resize(image, (target_size[1], target_size[0]), 
                         interpolation=cv2.INTER_AREA)
    print(f"✓ Image resized to: {resized.shape}")
    
    return resized

def convert_to_grayscale(image):
    """
    Convert color image to grayscale
    
    Args:
        image (ndarray): Input image (BGR format)
        
    Returns:
        ndarray: Grayscale image
    """
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(f"✓ Image converted to grayscale: {grayscale.shape}")
    
    return grayscale

def normalize_image(image):
    """
    Normalize image values to 0-1 range
    
    Args:
        image (ndarray): Input image
        
    Returns:
        ndarray: Normalized image
    """
    normalized = image.astype(np.float32) / 255.0
    return normalized

def process_image(image_path):
    """
    Complete preprocessing pipeline
    
    Args:
        image_path (str): Path to input image
        
    Returns:
        tuple: (BGR image, grayscale image)
    """
    print("\n--- TASK 2: IMAGE ACQUISITION & PREPROCESSING ---\n")
    
    # Load image
    original_image = load_image(image_path)
    
    # Resize to standard size
    original_image = resize_image(original_image, target_size=(512, 512))
    
    # Convert to grayscale
    grayscale_image = convert_to_grayscale(original_image)
    
    print("\nPreprocessing completed successfully!")
    
    return original_image, grayscale_image

def display_preprocessing(original_image, grayscale_image, output_dir="outputs"):
    """
    Display original and preprocessed images
    
    Args:
        original_image (ndarray): Original BGR image
        grayscale_image (ndarray): Grayscale image
        output_dir (str): Directory to save output
    """
    # Convert BGR to RGB for display
    original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('TASK 2: Image Acquisition & Preprocessing', 
                 fontsize=14, fontweight='bold')
    
    # Display original image
    axes[0].imshow(original_rgb)
    axes[0].set_title('Original Image (RGB)', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Display grayscale image
    axes[1].imshow(grayscale_image, cmap='gray')
    axes[1].set_title('Grayscale Image', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    output_path = utils.ensure_dir(output_dir)
    plt.savefig(f"{output_path}/task2_preprocessing.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Preprocessing visualization saved: task2_preprocessing.png")
    
    return fig

def display_image_stats(image, title="Image Statistics"):
    """
    Display image statistics
    
    Args:
        image (ndarray): Input image
        title (str): Title for statistics
    """
    print(f"\n{title}:")
    print(f"  Shape: {image.shape}")
    print(f"  Data type: {image.dtype}")
    print(f"  Min value: {image.min()}")
    print(f"  Max value: {image.max()}")
    print(f"  Mean value: {image.mean():.2f}")
    print(f"  Std deviation: {image.std():.2f}")

if __name__ == "__main__":
    # Test this module
    image_path = r"C:\Users\gtcam\OneDrive\Pictures\Camera Roll\OIP (2).webp"
    
    try:
        original, grayscale = process_image(image_path)
        
        display_image_stats(original, "Original Image Statistics")
        display_image_stats(grayscale, "Grayscale Image Statistics")
        
        display_preprocessing(original, grayscale)
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")
