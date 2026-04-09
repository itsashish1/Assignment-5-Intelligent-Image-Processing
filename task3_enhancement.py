"""
================================================================================
TASK 3: IMAGE ENHANCEMENT & RESTORATION
================================================================================

Purpose:
    - Simulate real-world degradation by adding noise
    - Apply restoration filters (Mean, Median, Gaussian)
    - Enhance contrast using Histogram Equalization and CLAHE
    - Compare enhanced and restored images

Delivered:
    - Degraded noisy images
    - Restored images using multiple filters
    - Enhanced images with improved contrast
    - Visual comparisons of all stages
================================================================================
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import exposure
import utils

def add_gaussian_noise(image, mean=0, sigma=25):
    """
    Add Gaussian noise to image
    
    Args:
        image (ndarray): Input image
        mean (float): Mean of Gaussian noise
        sigma (float): Standard deviation
        
    Returns:
        ndarray: Noisy image
    """
    noise = np.random.normal(mean, sigma, image.shape)
    noisy_image = np.clip(image.astype(float) + noise, 0, 255).astype(np.uint8)
    print(f"✓ Gaussian noise added (σ={sigma})")
    return noisy_image

def add_salt_pepper_noise(image, probability=0.05):
    """
    Add salt-and-pepper noise to image
    
    Args:
        image (ndarray): Input image
        probability (float): Probability of noise per pixel
        
    Returns:
        ndarray: Noisy image
    """
    noisy_image = image.copy()
    total_pixels = image.size
    
    # Salt (white pixels)
    num_salt = int(total_pixels * probability / 2)
    salt_coords = [np.random.randint(0, i, num_salt) for i in image.shape]
    noisy_image[tuple(salt_coords)] = 255
    
    # Pepper (black pixels)
    num_pepper = int(total_pixels * probability / 2)
    pepper_coords = [np.random.randint(0, i, num_pepper) for i in image.shape]
    noisy_image[tuple(pepper_coords)] = 0
    
    print(f"✓ Salt-and-pepper noise added (p={probability})")
    return noisy_image

def apply_mean_filter(image, kernel_size=5):
    """
    Apply mean (averaging) filter
    
    Args:
        image (ndarray): Input image
        kernel_size (int): Size of the filter kernel
        
    Returns:
        ndarray: Filtered image
    """
    filtered = cv2.blur(image, (kernel_size, kernel_size))
    print(f"✓ Mean filter applied (kernel={kernel_size}x{kernel_size})")
    return filtered

def apply_median_filter(image, kernel_size=5):
    """
    Apply median filter (excellent for salt-and-pepper noise)
    
    Args:
        image (ndarray): Input image
        kernel_size (int): Size of the filter kernel
        
    Returns:
        ndarray: Filtered image
    """
    filtered = cv2.medianBlur(image, kernel_size)
    print(f"✓ Median filter applied (kernel={kernel_size}x{kernel_size})")
    return filtered

def apply_gaussian_filter(image, sigma=1.0):
    """
    Apply Gaussian filter
    
    Args:
        image (ndarray): Input image
        sigma (float): Standard deviation for Gaussian kernel
        
    Returns:
        ndarray: Filtered image
    """
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    filtered = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    print(f"✓ Gaussian filter applied (σ={sigma})")
    return filtered

def apply_histogram_equalization(image):
    """
    Apply histogram equalization for contrast enhancement
    
    Args:
        image (ndarray): Input image
        
    Returns:
        ndarray: Enhanced image
    """
    enhanced = cv2.equalizeHist(image)
    print(f"✓ Histogram Equalization applied")
    return enhanced

def apply_clahe(image, clip_limit=2.0, tile_size=8):
    """
    Apply Contrast Limited Adaptive Histogram Equalization
    
    Args:
        image (ndarray): Input image
        clip_limit (float): Clip limit for CLAHE
        tile_size (int): Size of grid tiles
        
    Returns:
        ndarray: Enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    enhanced = clahe.apply(image)
    print(f"✓ CLAHE applied (clip_limit={clip_limit}, tile_size={tile_size})")
    return enhanced

def enhance_and_restore(grayscale_image, output_dir="outputs"):
    """
    Complete enhancement and restoration pipeline
    
    Args:
        grayscale_image (ndarray): Input grayscale image
        output_dir (str): Directory to save outputs
        
    Returns:
        tuple: (noisy_image, restored_images_dict, enhanced_image)
    """
    print("\n--- TASK 3: IMAGE ENHANCEMENT & RESTORATION ---\n")
    
    # Step 1: Create degraded image with noise
    print("Step 1: Simulating real-world degradation...")
    noisy_image = add_gaussian_noise(grayscale_image, sigma=25)
    noisy_image = add_salt_pepper_noise(noisy_image, probability=0.05)
    
    # Step 2: Apply restoration filters
    print("\nStep 2: Applying restoration filters...")
    mean_filtered = apply_mean_filter(noisy_image, kernel_size=5)
    median_filtered = apply_median_filter(noisy_image, kernel_size=5)
    gaussian_filtered = apply_gaussian_filter(noisy_image, sigma=1.5)
    
    restored_images = {
        'mean': mean_filtered,
        'median': median_filtered,
        'gaussian': gaussian_filtered
    }
    
    # Step 3: Apply contrast enhancement on best restoration result
    print("\nStep 3: Applying contrast enhancement...")
    # Use median filter result as it's best for salt-and-pepper noise
    global_hist = apply_histogram_equalization(median_filtered)
    clahe_enhanced = apply_clahe(median_filtered, clip_limit=3.0, tile_size=8)
    
    # Use CLAHE as final enhanced image (better detail preservation)
    enhanced_image = clahe_enhanced
    
    # Visualize restoration process
    display_restoration(grayscale_image, noisy_image, mean_filtered, 
                       median_filtered, gaussian_filtered, enhanced_image, output_dir)
    
    print("\nEnhancement and restoration completed!")
    
    return noisy_image, restored_images, enhanced_image

def display_restoration(original, noisy, mean_f, median_f, gaussian_f, enhanced, output_dir="outputs"):
    """
    Display restoration process comparison
    
    Args:
        original (ndarray): Original image
        noisy (ndarray): Noisy image
        mean_f (ndarray): Mean filtered image
        median_f (ndarray): Median filtered image
        gaussian_f (ndarray): Gaussian filtered image
        enhanced (ndarray): Final enhanced image
        output_dir (str): Output directory
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('TASK 3: Image Enhancement & Restoration Pipeline', 
                 fontsize=14, fontweight='bold')
    
    images = [original, noisy, mean_f, median_f, gaussian_f, enhanced]
    titles = ['Original', 'Noisy (Gaussian + S&P)', 
              'Mean Filter', 'Median Filter', 
              'Gaussian Filter', 'Final Enhanced (CLAHE)']
    
    for idx, (ax, img, title) in enumerate(zip(axes.flat, images, titles)):
        ax.imshow(img, cmap='gray')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save figure
    output_path = utils.ensure_dir(output_dir)
    plt.savefig(f"{output_path}/task3_enhancement.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Enhancement visualization saved: task3_enhancement.png")

if __name__ == "__main__":
    import task2_preprocessing as task2
    
    image_path = r"C:\Users\gtcam\OneDrive\Pictures\Camera Roll\OIP (2).webp"
    
    try:
        _, grayscale = task2.process_image(image_path)
        noisy, restored, enhanced = enhance_and_restore(grayscale)
        plt.show()
    except Exception as e:
        print(f"Error: {e}")
