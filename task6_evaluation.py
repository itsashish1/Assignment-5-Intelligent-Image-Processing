"""
================================================================================
TASK 6: PERFORMANCE EVALUATION
================================================================================

Purpose:
    - Compute image quality metrics:
      * Mean Squared Error (MSE)
      * Peak Signal-to-Noise Ratio (PSNR)
      * Structural Similarity Index (SSIM)
    - Compare original vs enhanced and restored images
    - Provide quantitative performance assessment

Delivered:
    - Computed evaluation metrics
    - Performance comparison analysis
    - Metric interpretation and conclusions
================================================================================
"""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import utils

def compute_mse(image1, image2):
    """
    Compute Mean Squared Error between two images
    
    Args:
        image1 (ndarray): First image
        image2 (ndarray): Second image
        
    Returns:
        float: MSE value
    """
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same shape")
    
    # Convert to float for computation
    img1 = image1.astype(np.float32)
    img2 = image2.astype(np.float32)
    
    mse = np.mean((img1 - img2) ** 2)
    return mse

def compute_psnr(image1, image2, max_pixel_value=255):
    """
    Compute Peak Signal-to-Noise Ratio
    
    Args:
        image1 (ndarray): First image (reference)
        image2 (ndarray): Second image
        max_pixel_value (int): Maximum pixel value
        
    Returns:
        float: PSNR value in dB
    """
    mse = compute_mse(image1, image2)
    
    if mse == 0:
        return float('inf')
    
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr

def compute_ssim(image1, image2, max_pixel_value=255):
    """
    Compute Structural Similarity Index (SSIM)
    
    Args:
        image1 (ndarray): First image
        image2 (ndarray): Second image
        max_pixel_value (int): Maximum pixel value
        
    Returns:
        float: SSIM value (range: -1 to 1, where 1 is identical)
    """
    # Convert to float
    img1 = image1.astype(np.float32)
    img2 = image2.astype(np.float32)
    
    # Constants for SSIM
    c1 = (0.01 * max_pixel_value) ** 2
    c2 = (0.03 * max_pixel_value) ** 2
    
    # Compute mean
    mu1 = gaussian_filter(img1, sigma=1.5)
    mu2 = gaussian_filter(img2, sigma=1.5)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Compute variance and covariance
    sigma1_sq = gaussian_filter(img1 ** 2, sigma=1.5) - mu1_sq
    sigma2_sq = gaussian_filter(img2 ** 2, sigma=1.5) - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, sigma=1.5) - mu1_mu2
    
    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    
    # Return mean SSIM
    return np.mean(ssim_map)

def evaluate_performance(original, grayscale, noisy, restored, enhanced):
    """
    Complete performance evaluation pipeline
    
    Args:
        original (ndarray): Original BGR image
        grayscale (ndarray): Grayscale image
        noisy (ndarray): Noisy grayscale image
        restored (dict): Dictionary of restored images
        enhanced (ndarray): Enhanced image
        
    Returns:
        dict: Dictionary of computed metrics
    """
    print("\n--- TASK 6: PERFORMANCE EVALUATION ---\n")
    
    # Convert original to grayscale for fair comparison
    if len(original.shape) == 3:
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = original
    
    print("Computing image quality metrics...\n")
    
    metrics = {}
    
    # ===== Comparison 1: Original vs Noisy =====
    print("=" * 70)
    print("COMPARISON 1: ORIGINAL vs NOISY IMAGE")
    print("=" * 70)
    
    mse_noisy = compute_mse(original_gray, noisy)
    psnr_noisy = compute_psnr(original_gray, noisy)
    ssim_noisy = compute_ssim(original_gray, noisy)
    
    metrics['original_vs_noisy'] = {
        'MSE': mse_noisy,
        'PSNR': psnr_noisy,
        'SSIM': ssim_noisy
    }
    
    print(f"\nMetric                  Value")
    print("-" * 70)
    print(f"Mean Squared Error:     {mse_noisy:.4f}")
    print(f"PSNR (dB):             {psnr_noisy:.4f}")
    print(f"SSIM:                  {ssim_noisy:.4f}")
    print(f"\nInterpretation:")
    print(f"  • High MSE indicates significant degradation due to noise")
    print(f"  • Lower PSNR means more noise relative to signal")
    print(f"  • Lower SSIM shows structural dissimilarity")
    
    # ===== Comparison 2: Original vs Restored (Median) =====
    print("\n" + "=" * 70)
    print("COMPARISON 2: ORIGINAL vs RESTORED IMAGE (Median Filter)")
    print("=" * 70)
    
    median_restored = restored['median']
    mse_restored = compute_mse(original_gray, median_restored)
    psnr_restored = compute_psnr(original_gray, median_restored)
    ssim_restored = compute_ssim(original_gray, median_restored)
    
    metrics['original_vs_restored'] = {
        'MSE': mse_restored,
        'PSNR': psnr_restored,
        'SSIM': ssim_restored
    }
    
    print(f"\nMetric                  Value")
    print("-" * 70)
    print(f"Mean Squared Error:     {mse_restored:.4f}")
    print(f"PSNR (dB):             {psnr_restored:.4f}")
    print(f"SSIM:                  {ssim_restored:.4f}")
    print(f"\nImprovement over Noisy:")
    print(f"  • MSE reduction:       {((mse_noisy - mse_restored) / mse_noisy * 100):.2f}%")
    print(f"  • PSNR improvement:    {(psnr_restored - psnr_noisy):.4f} dB")
    print(f"  • SSIM improvement:    {(ssim_restored - ssim_noisy):.4f}")
    
    # ===== Comparison 3: Original vs Enhanced =====
    print("\n" + "=" * 70)
    print("COMPARISON 3: ORIGINAL vs ENHANCED IMAGE (CLAHE)")
    print("=" * 70)
    
    mse_enhanced = compute_mse(original_gray, enhanced)
    psnr_enhanced = compute_psnr(original_gray, enhanced)
    ssim_enhanced = compute_ssim(original_gray, enhanced)
    
    metrics['original_vs_enhanced'] = {
        'MSE': mse_enhanced,
        'PSNR': psnr_enhanced,
        'SSIM': ssim_enhanced
    }
    
    print(f"\nMetric                  Value")
    print("-" * 70)
    print(f"Mean Squared Error:     {mse_enhanced:.4f}")
    print(f"PSNR (dB):             {psnr_enhanced:.4f}")
    print(f"SSIM:                  {ssim_enhanced:.4f}")
    print(f"\nInterpretation:")
    print(f"  • Enhancement may have different pixel values but preserves structure")
    print(f"  • Focus on SSIM for perceptual quality rather than pixel-level MSE")
    
    # ===== Comparison 4: Noisy vs Restored =====
    print("\n" + "=" * 70)
    print("COMPARISON 4: NOISY vs RESTORED IMAGE (Noise Removal Performance)")
    print("=" * 70)
    
    mse_removal = compute_mse(noisy, median_restored)
    psnr_removal = compute_psnr(noisy, median_restored)
    ssim_removal = compute_ssim(noisy, median_restored)
    
    metrics['noisy_vs_restored'] = {
        'MSE': mse_removal,
        'PSNR': psnr_removal,
        'SSIM': ssim_removal
    }
    
    print(f"\nMetric                  Value")
    print("-" * 70)
    print(f"Mean Squared Error:     {mse_removal:.4f}")
    print(f"PSNR (dB):             {psnr_removal:.4f}")
    print(f"SSIM:                  {ssim_removal:.4f}")
    print(f"\nInterpretation:")
    print(f"  • Lower values indicate effective noise removal")
    print(f"  • High PSNR and SSIM suggest good restoration quality")
    
    # ===== Summary =====
    print("\n" + "=" * 70)
    print("SUMMARY OF METRICS")
    print("=" * 70)
    
    print("\nMetric Ranges:")
    print("  • MSE: Lower is better (minimum = 0 for identical images)")
    print("  • PSNR (dB): Higher is better (typical range 10-50+ dB)")
    print("  • SSIM: Higher is better (range -1 to 1, where 1 is identical)")
    
    print("\nKey Findings:")
    print(f"  1. Restoration Quality: PSNR improvement of {(psnr_restored - psnr_noisy):.2f} dB")
    print(f"  2. Noise Reduction: MSE reduced by {((mse_noisy - mse_restored) / mse_noisy * 100):.2f}%")
    print(f"  3. Structural Similarity: SSIM = {ssim_restored:.4f} after restoration")
    print(f"  4. Enhancement Effect: SSIM = {ssim_enhanced:.4f} after enhancement")
    
    print("\n" + "=" * 70)
    
    return metrics

def plot_metrics(metrics):
    """
    Plot performance metrics comparison
    
    Args:
        metrics (dict): Dictionary of computed metrics
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('TASK 6: Performance Evaluation Metrics', 
                 fontsize=14, fontweight='bold')
    
    comparisons = list(metrics.keys())
    mse_values = [metrics[c]['MSE'] for c in comparisons]
    psnr_values = [metrics[c]['PSNR'] for c in comparisons]
    ssim_values = [metrics[c]['SSIM'] for c in comparisons]
    
    # Labels for x-axis
    labels = ['Orig vs\nNoisy', 'Orig vs\nRestored', 'Orig vs\nEnhanced', 'Noisy vs\nRestored']
    labels = labels[:len(comparisons)]
    
    # MSE plot
    axes[0].bar(labels, mse_values, color='#FF6B6B', alpha=0.8)
    axes[0].set_title('Mean Squared Error (MSE)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('MSE Value')
    axes[0].grid(axis='y', alpha=0.3)
    
    # PSNR plot
    axes[1].bar(labels, psnr_values, color='#4ECDC4', alpha=0.8)
    axes[1].set_title('Peak Signal-to-Noise Ratio (PSNR)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('PSNR (dB)')
    axes[1].grid(axis='y', alpha=0.3)
    
    # SSIM plot
    axes[2].bar(labels, ssim_values, color='#45B7D1', alpha=0.8)
    axes[2].set_title('Structural Similarity Index (SSIM)', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('SSIM Value')
    axes[2].set_ylim([0, 1])
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig
