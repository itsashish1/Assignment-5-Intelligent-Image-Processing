"""
================================================================================
UTILITY FUNCTIONS
================================================================================

Purpose:
    - Provide common utility functions for all tasks
    - Directory management, image I/O, visualization helpers
================================================================================
"""

import os
import cv2
import numpy as np
from pathlib import Path

def ensure_dir(directory):
    """
    Create directory if it doesn't exist
    
    Args:
        directory (str): Directory path
        
    Returns:
        str: Directory path
    """
    Path(directory).mkdir(parents=True, exist_ok=True)
    return directory

def save_image(image, filename, output_dir="outputs"):
    """
    Save image to file
    
    Args:
        image (ndarray): Image to save
        filename (str): Filename to save as
        output_dir (str): Output directory
    """
    output_path = ensure_dir(output_dir)
    full_path = f"{output_path}/{filename}"
    
    if len(image.shape) == 2:  # Grayscale
        cv2.imwrite(full_path, image)
    else:  # Color
        cv2.imwrite(full_path, image)
    
    print(f"✓ Image saved: {filename}")
    return full_path

def load_image(image_path):
    """
    Load Image from file
    
    Args:
        image_path (str): Path to image file
        
    Returns:
        ndarray: Loaded image
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return image

def get_image_info(image):
    """
    Get image information
    
    Args:
        image (ndarray): Image
        
    Returns:
        dict: Image information
    """
    info = {
        'shape': image.shape,
        'dtype': image.dtype,
        'size': image.size,
        'min': image.min(),
        'max': image.max(),
        'mean': image.mean()
    }
    return info

def normalize_image(image, target_min=0, target_max=255):
    """
    Normalize image to target range
    
    Args:
        image (ndarray): Input image
        target_min (float): Target minimum value
        target_max (float): Target maximum value
        
    Returns:
        ndarray: Normalized image
    """
    image_min = image.min()
    image_max = image.max()
    
    normalized = (image - image_min) / (image_max - image_min)
    normalized = normalized * (target_max - target_min) + target_min
    
    return normalized.astype(np.uint8)

def to_uint8(image):
    """
    Convert image to uint8 format
    
    Args:
        image (ndarray): Input image
        
    Returns:
        ndarray: uint8 image
    """
    if image.dtype == np.uint8:
        return image
    
    if image.max() <= 1.0:
        return (image * 255).astype(np.uint8)
    else:
        return np.clip(image, 0, 255).astype(np.uint8)

def to_float32(image):
    """
    Convert image to float32 format (0-1 range)
    
    Args:
        image (ndarray): Input image
        
    Returns:
        ndarray: float32 image
    """
    if image.dtype == np.float32:
        return image
    
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    else:
        normalized = (image - image.min()) / (image.max() - image.min())
        return normalized.astype(np.float32)
