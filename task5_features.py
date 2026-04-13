"""
================================================================================
TASK 5: OBJECT REPRESENTATION & FEATURE EXTRACTION
================================================================================

Purpose:
    - Detect edges using Sobel and Canny edge detectors
    - Extract object contours and draw bounding boxes
    - Extract features using ORB (Oriented FAST and Rotated BRIEF)
    - Visualize keypoints and feature descriptors
    - Represent objects for recognition and tracking

Delivered:
    - Edge detection maps (Sobel, Canny)
    - Contour extraction and bounding boxes
    - ORB keypoints and descriptors
    - Feature visualization
================================================================================
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils

def apply_sobel_operator(image):
    """
    Apply Sobel edge detector
    
    Args:
        image (ndarray): Input grayscale image
        
    Returns:
        ndarray: Sobel edge map
    """
    # Compute gradients in X and Y directions
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute magnitude of gradients
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    magnitude = np.uint8(np.clip(magnitude, 0, 255))
    
    print(f"* Sobel operator applied")
    return magnitude

def apply_canny_detector(image, threshold1=50, threshold2=150):
    """
    Apply Canny edge detector
    
    Args:
        image (ndarray): Input grayscale image
        threshold1 (int): Lower threshold
        threshold2 (int): Upper threshold
        
    Returns:
        ndarray: Canny edge map
    """
    edges = cv2.Canny(image, threshold1, threshold2)
    print(f"* Canny edge detector applied (thresholds: {threshold1}, {threshold2})")
    return edges

def extract_contours(binary_image):
    """
    Extract contours from binary image
    
    Args:
        binary_image (ndarray): Binary image
        
    Returns:
        list: List of contours
    """
    contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(f"* Found {len(contours)} contours")
    return contours

def draw_bounding_boxes(image, contours, min_area=500):
    """
    Draw bounding boxes around contours
    
    Args:
        image (ndarray): Input image (will be converted to RGB for visualization)
        contours (list): List of contours
        min_area (float): Minimum contour area to draw bounding box
        
    Returns:
        ndarray: Image with bounding boxes
    """
    result = image.copy()
    
    # Convert grayscale to BGR for colored boxes
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    box_count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
            box_count += 1
    
    print(f"* Drew {box_count} bounding boxes")
    return result

def extract_orb_features(image):
    """
    Extract ORB features (Oriented FAST and Rotated BRIEF)
    
    Args:
        image (ndarray): Input grayscale image
        
    Returns:
        tuple: (keypoints, descriptors)
    """
    # Create ORB detector
    orb = cv2.ORB_create(nfeatures=500)
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(image, None)
    
    print(f"* ORB features extracted: {len(keypoints)} keypoints")
    return keypoints, descriptors

def draw_keypoints(image, keypoints):
    """
    Draw keypoints on image
    
    Args:
        image (ndarray): Input image
        keypoints (list): List of keypoints
        
    Returns:
        ndarray: Image with drawn keypoints
    """
    result = cv2.drawKeypoints(image, keypoints, None, 
                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    print(f"* Drew {len(keypoints)} keypoints")
    return result

def extract_features(original_image, enhanced_image, output_dir="outputs"):
    """
    Complete feature extraction pipeline
    
    Args:
        original_image (ndarray): Original RGB image
        enhanced_image (ndarray): Enhanced grayscale image
        output_dir (str): Output directory
        
    Returns:
        tuple: (edge_images_dict, featured_image)
    """
    print("\n--- TASK 5: OBJECT REPRESENTATION & FEATURE EXTRACTION ---\n")
    
    print("Step 1: Edge detection...")
    
    # Convert original to grayscale if needed
    if len(original_image.shape) == 3:
        original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = original_image
    
    # Apply edge detectors
    sobel_edges = apply_sobel_operator(enhanced_image)
    canny_edges = apply_canny_detector(enhanced_image, threshold1=50, threshold2=150)
    
    print("\nStep 2: Contour extraction...")
    
    # Extract contours from Canny edges
    contours = extract_contours(canny_edges)
    
    # Draw bounding boxes on original image
    image_with_boxes = draw_bounding_boxes(original_gray, contours, min_area=500)
    
    print("\nStep 3: Feature extraction...")
    
    # Extract ORB features from original image
    keypoints, descriptors = extract_orb_features(original_gray)
    
    # Draw keypoints
    featured_image = draw_keypoints(original_gray, keypoints)
    
    edge_images = {
        'sobel': sobel_edges,
        'canny': canny_edges,
        'bounding_boxes': image_with_boxes
    }
    
    # Visualize edge detection and features
    display_features(sobel_edges, canny_edges, image_with_boxes, featured_image, output_dir)
    
    print("\nFeature extraction completed!")
    
    return edge_images, featured_image

def display_features(sobel, canny, boxes, featured, output_dir="outputs"):
    """
    Display edge detection and feature extraction results
    
    Args:
        sobel (ndarray): Sobel edge map
        canny (ndarray): Canny edge map
        boxes (ndarray): Image with bounding boxes
        featured (ndarray): Image with keypoints
        output_dir (str): Output directory
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('TASK 5: Object Representation & Feature Extraction', 
                 fontsize=14, fontweight='bold')
    
    # Sobel edges
    axes[0, 0].imshow(sobel, cmap='gray')
    axes[0, 0].set_title('Sobel Edge Detector', fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Canny edges
    axes[0, 1].imshow(canny, cmap='gray')
    axes[0, 1].set_title('Canny Edge Detector', fontsize=11, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Bounding boxes
    if len(boxes.shape) == 3 and boxes.shape[2] == 3:
        boxes_rgb = cv2.cvtColor(boxes, cv2.COLOR_BGR2RGB)
        axes[1, 0].imshow(boxes_rgb)
    else:
        axes[1, 0].imshow(boxes, cmap='gray')
    axes[1, 0].set_title('Contours with Bounding Boxes', fontsize=11, fontweight='bold')
    axes[1, 0].axis('off')
    
    # ORB keypoints
    axes[1, 1].imshow(featured, cmap='gray')
    axes[1, 1].set_title('ORB Keypoints Detection', fontsize=11, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    output_path = utils.ensure_dir(output_dir)
    plt.savefig(f"{output_path}/task5_features.png", dpi=150, bbox_inches='tight')
    print(f"\n* Feature extraction visualization saved: task5_features.png")
