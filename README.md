# Intelligent Image Processing System - Assignment-5

![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Project Overview

A comprehensive **end-to-end image processing pipeline** demonstrating advanced computer vision techniques including acquisition, preprocessing, enhancement, segmentation, feature extraction, performance evaluation, and visualization.

**Student Details:**
- **Name:** Ashish Yadav
- **Roll No:** 2301010413
- **Course:** BTech CS (Computer Science & Engineering)
- **Assignment:** Assignment-5 - Intelligent Image Processing System
- **Date:** April 9, 2026

---

## 🎯 System Capabilities

### ✓ Task 1: Project Setup & System Overview
- Structured project folder with clear organization
- Python script with comprehensive header comments
- Welcome message describing system purpose and capabilities

### ✓ Task 2: Image Acquisition & Preprocessing
- **Image Loading:** Support for JPEG, PNG, WebP formats
- **Resizing:** Standardized to 512×512 resolution
- **Color Conversion:** RGB to Grayscale transformation
- **Visualization:** Original vs Preprocessed comparison

### ✓ Task 3: Image Enhancement & Restoration
- **Noise Simulation:**
  - Gaussian noise (σ=25)
  - Salt-and-pepper noise (p=0.05)
- **Restoration Filters:**
  - Mean filter (5×5 kernel)
  - Median filter (5×5 kernel)
  - Gaussian filter (σ=1.5)
- **Contrast Enhancement:**
  - Histogram Equalization
  - CLAHE (Contrast Limited Adaptive Histogram Equalization)

### ✓ Task 4: Image Segmentation & Morphological Processing
- **Thresholding Techniques:**
  - Global thresholding (fixed threshold=127)
  - Otsu's automatic thresholding
  - Adaptive thresholding (Gaussian)
- **Morphological Operations:**
  - Dilation (expands objects)
  - Erosion (shrinks objects)
  - Opening (removes small objects)
  - Closing (fills holes)
  - Morphological gradient (edge extraction)

### ✓ Task 5: Object Representation & Feature Extraction
- **Edge Detection:**
  - Sobel operator (gradient-based)
  - Canny edge detector (multi-stage)
- **Contour Analysis:**
  - Extracted 3,174 contours
  - Drew 31 bounding boxes
- **Feature Extraction:**
  - ORB detector: 498 keypoints extracted
  - Rotation and scale-invariant features
  - Keypoint visualization

### ✓ Task 6: Performance Evaluation
**Quantitative Metrics:**
- **MSE (Mean Squared Error)**: Pixel-level differences
- **PSNR (Peak Signal-to-Noise Ratio)**: Signal quality in dB
- **SSIM (Structural Similarity Index)**: Perceptual similarity (0-1)

**Key Results:**
- MSE reduction: **96.56%** (1567.67 → 53.86)
- PSNR improvement: **+14.64 dB** (16.18 → 30.82 dB)
- SSIM improvement: **+0.664** (0.088 → 0.732)

### ✓ Task 7: Final Visualization & Analysis
- Complete 6-stage pipeline visualization
- Original → Noisy → Restored → Enhanced → Segmented → Featured
- Comprehensive system performance conclusions
- Recommendations for future enhancements

---

## 📦 Project Structure

```
intelligent_image_system/
├── main.py                    # Main execution script
├── task1_setup.py             # Task 1: Project setup & welcome
├── task2_preprocessing.py      # Task 2: Image acquisition & preprocessing
├── task3_enhancement.py        # Task 3: Enhancement & restoration
├── task4_segmentation.py       # Task 4: Segmentation & morphology
├── task5_features.py           # Task 5: Feature extraction
├── task6_evaluation.py         # Task 6: Performance evaluation
├── task7_visualization.py      # Task 7: Final visualization & analysis
├── utils.py                    # Utility functions
├── README.md                   # Project documentation
├── .gitignore                  # Git ignore file
└── outputs/                    # Generated visualizations
    ├── task2_preprocessing.png
    ├── task3_enhancement.png
    ├── task4_segmentation.png
    ├── task5_features.png
    └── task7_pipeline_visualization.png
```

---

## 🔧 Installation & Setup

### Prerequisites
- Python 3.7+
- pip or conda package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/Assignment-5-Intelligent-Image-Processing.git
cd intelligent_image_system
```

### Step 2: Create Virtual Environment (Optional but Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Required Packages:**
```
opencv-python>=4.5.0
numpy>=1.21.0
matplotlib>=3.4.0
scipy>=1.7.0
scikit-image>=0.18.0
scikit-learn>=0.24.0
Pillow>=8.3.0
```

---

## 🚀 Usage

### Run the Complete Pipeline
```bash
python main.py
```

This will execute all 7 tasks sequentially and generate visualizations in the `outputs/` folder.

### Run Individual Tasks
```python
import task2_preprocessing as task2
import task3_enhancement as task3

# Load and preprocess image
original, grayscale = task2.process_image("path/to/image.jpg")

# Enhance and restore
noisy, restored, enhanced = task3.enhance_and_restore(grayscale)
```

---

## 📊 Output Results

### Performance Metrics Summary

| Comparison | MSE | PSNR (dB) | SSIM |
|-----------|-----|-----------|------|
| Original vs Noisy | 1567.67 | 16.18 | 0.0880 |
| Original vs Restored | 53.86 | 30.82 | 0.7320 |
| Original vs Enhanced | 1357.38 | 16.80 | 0.3119 |
| Noisy vs Restored | 1546.16 | 16.24 | 0.0998 |

### Generated Visualizations

1. **task2_preprocessing.png** - Original vs Grayscale images
2. **task3_enhancement.png** - Restoration pipeline (6 stages)
3. **task4_segmentation.png** - Segmentation & morphology (6 results)
4. **task5_features.png** - Edge detection & features (4 methods)
5. **task7_pipeline_visualization.png** - Complete 6-stage pipeline

---

## 🔬 Technical Details

### Libraries Used

| Library | Purpose |
|---------|---------|
| **OpenCV (cv2)** | Image processing & computer vision algorithms |
| **NumPy** | Numerical computations & matrix operations |
| **SciPy** | Scientific computing & signal processing |
| **Matplotlib** | Visualization & plotting |
| **Scikit-Image** | Advanced image processing algorithms |
| **Pillow** | Image file I/O operations |

### Key Algorithms Implemented

1. **Noise Addition**
   - Gaussian noise: `N(μ=0, σ=25)`
   - Salt-and-pepper: 5% pixel modification

2. **Filtering**
   - Mean filter: Simple averaging
   - Median filter: Order statistics
   - Gaussian filter: Gaussian kernel convolution

3. **Enhancement**
   - Histogram equalization: Global contrast
   - CLAHE: Adaptive local contrast (tile_size=8, clip_limit=3.0)

4. **Segmentation**
   - Otsu's method: Automatic threshold selection
   - Adaptive: Gaussian-weighted thresholding

5. **Morphology**
   - Dilation/Erosion: Max/min filtering with structuring element
   - Opening/Closing: Composite operations

6. **Feature Extraction**
   - ORB: Fast rotation-invariant features
   - Sobel: Gradient-based edge detection
   - Canny: Multi-stage edge detection

### Performance Results

**Noise Reduction Effectiveness:**
- Median filter removed **96.56%** of noise (MSE reduction)
- PSNR improved by **14.64 dB** over noisy image
- Structural similarity (SSIM) restored to **0.732** (good quality)

---

## 📈 System Performance

### Execution Time
- Image loading & preprocessing: ~0.5s
- Noise simulation: ~0.2s
- Restoration & enhancement: ~1.0s
- Segmentation: ~0.5s
- Feature extraction: ~0.3s
- Evaluation: ~2.0s
- Visualization: ~1.5s
- **Total: ~6 seconds**

### Memory Usage
- Image storage: ~1-2 MB (512×512)
- Processing buffers: ~5-10 MB
- Visualizations: ~10-15 MB
- **Peak: ~25 MB**

---

## 🎓 Learning Outcomes

This project demonstrates:
- **Image I/O & Preprocessing**: Loading, resizing, color space conversion
- **Noise Modeling**: Simulation of real-world degradation
- **Signal Processing**: Filtering and enhancement techniques
- **Computer Vision**: Segmentation and feature extraction
- **Performance Analysis**: Quantitative metric computation
- **Data Visualization**: Multi-stage pipeline display
- **Software Engineering**: Modular code organization and documentation

---

## 🚀 Future Enhancements

1. **GPU Acceleration**
   - CUDA implementation for large batches
   - Real-time processing capability

2. **Advanced Techniques**
   - Deep learning-based denoising (autoencoders)
   - CNN-based segmentation (U-Net, Mask R-CNN)
   - Transfer learning for feature extraction

3. **Additional Features**
   - Video processing pipeline
   - Multi-scale processing
   - Parallel processing support

4. **Production Deployment**
   - REST API for image processing
   - Web interface with Flask/Django
   - Docker containerization

---

## 📝 Example Output

```
╔════════════════════════════════════════════════════════════════╗
║     INTELLIGENT IMAGE PROCESSING SYSTEM - WELCOME              ║
╚════════════════════════════════════════════════════════════════╝

Student Name:    Ashish Yadav
Roll No:         2301010413
Course:          BTech CSE
Assignment:      Assignment-5
Date:            April 09, 2026

======================================================================
TASK 1: PROJECT SETUP & SYSTEM OVERVIEW ✓
======================================================================

TASK 2: IMAGE ACQUISITION & PREPROCESSING
...✓ Image loaded: 768×329 pixels
...✓ Image resized to: 512×512 pixels
...✓ Preprocessing visualization saved

TASK 3: IMAGE ENHANCEMENT & RESTORATION
...✓ Gaussian noise added (σ=25)
...✓ Median filter applied
...✓ CLAHE applied (enhancement)

[continues for all tasks...]

✓ ALL TASKS COMPLETED SUCCESSFULLY!
```

---

## 📚 References

### Computer Vision Fundamentals
- Sobel, I. (1968). "Gradient-based edge detection"
- Canny, J. (1986). "A Computational Approach to Edge Detection"
- Otsu, N. (1979). "A Threshold Selection Method"

### Image Processing Techniques
- Gonzalez & Woods. "Digital Image Processing" (3rd Ed.)
- OpenCV Documentation: https://docs.opencv.org/

### Metrics & Evaluation
- Wang & Bovik. "Mean squared error: Love it or leave it?"
- Wang et al. "Image Quality Assessment: From Error Visibility to Structural Similarity"

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👨‍💼 Author

**Ashish Yadav**
- Roll No: 2301010413
- Course: BTech CS (Computer Science & Engineering)
- Email: ashish.yadav@college.edu

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## ❓ FAQ

**Q: Can I use my own images?**
A: Yes! Modify the `image_path` variable in `main.py` to point to your image file.

**Q: What image formats are supported?**
A: JPEG, PNG, WebP, BMP, TIFF, and most common formats supported by OpenCV.

**Q: How can I adjust parameters?**
A: Each task module has adjustable parameters (kernel sizes, thresholds, etc.) that can be modified.

**Q: Can this work on GPU?**
A: Currently CPU-based. GPU acceleration can be added using CUDA/CuPy.

---

## 📧 Contact & Support

For issues, questions, or suggestions:
- Create an issue on GitHub
- Email: ashish.yadav@college.edu
- Subject: Assignment-5 Support

---

**Last Updated:** April 9, 2026
**Version:** 1.0.0
