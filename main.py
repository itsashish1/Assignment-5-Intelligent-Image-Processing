"""
================================================================================
        INTELLIGENT IMAGE PROCESSING SYSTEM
================================================================================

Student Name:       Ashish Yadav
Roll No:            2301010413
Course Name:        BTech CSE (Computer Science & Engineering)
Assignment Title:   Assignment-5 - Intelligent Image Processing System
Date:               April 9, 2026

================================================================================
PURPOSE:
This system demonstrates a complete pipeline for intelligent image processing
including acquisition, preprocessing, enhancement, segmentation, feature
extraction, evaluation, and visualization.
================================================================================
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import all task modules
import task1_setup as task1
import task2_preprocessing as task2
import task3_enhancement as task3
import task4_segmentation as task4
import task5_features as task5
import task6_evaluation as task6
import task7_visualization as task7
import utils

def main():
    """Main execution function"""
    
    # Display welcome message
    task1.welcome_message()
    
    # Image path
    image_path = r"C:\Users\gtcam\OneDrive\Pictures\Camera Roll\OIP (2).webp"
    output_dir = "outputs"
    
    try:
        # TASK 1: Project Setup (Already done via task1.py)
        print("\n" + "="*70)
        print("TASK 1: PROJECT SETUP & SYSTEM OVERVIEW ✓")
        print("="*70)
        
        # TASK 2: Image Acquisition & Preprocessing
        print("\nTASK 2: IMAGE ACQUISITION & PREPROCESSING")
        print("-"*70)
        original_image, grayscale_image = task2.process_image(image_path)
        task2.display_preprocessing(original_image, grayscale_image, output_dir)
        
        # TASK 3: Image Enhancement & Restoration
        print("\nTASK 3: IMAGE ENHANCEMENT & RESTORATION")
        print("-"*70)
        noisy_image, restored_images, enhanced_image = task3.enhance_and_restore(
            grayscale_image, output_dir
        )
        
        # TASK 4: Image Segmentation & Morphological Processing
        print("\nTASK 4: IMAGE SEGMENTATION & MORPHOLOGICAL PROCESSING")
        print("-"*70)
        segmented_images = task4.segment_and_morph(enhanced_image, output_dir)
        
        # TASK 5: Object Representation & Feature Extraction
        print("\nTASK 5: OBJECT REPRESENTATION & FEATURE EXTRACTION")
        print("-"*70)
        edge_images, featured_image = task5.extract_features(
            original_image, enhanced_image, output_dir
        )
        
        # TASK 6: Performance Evaluation
        print("\nTASK 6: PERFORMANCE EVALUATION")
        print("-"*70)
        metrics = task6.evaluate_performance(
            original_image, grayscale_image, noisy_image, 
            restored_images, enhanced_image
        )
        
        # TASK 7: Final Visualization & Analysis
        print("\nTASK 7: FINAL VISUALIZATION & ANALYSIS")
        print("-"*70)
        task7.final_visualization(
            original_image, grayscale_image, noisy_image,
            restored_images, enhanced_image, segmented_images,
            featured_image, output_dir
        )
        
        print("\n" + "="*70)
        print("✓ ALL TASKS COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"\nOutput files saved in: {output_dir}/")
        print("\nGenerated files:")
        print("  - task2_preprocessing.png (Original & Grayscale)")
        print("  - task3_enhancement.png (Restoration comparison)")
        print("  - task4_segmentation.png (Segmented images)")
        print("  - task5_features.png (Edge detection & features)")
        print("  - task7_final_visualization.png (Complete pipeline)")
        
    except FileNotFoundError:
        print(f"ERROR: Image file not found at {image_path}")
        print("Please ensure the image file exists at the specified path.")
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
    plt.show()
