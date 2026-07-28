# Intelligent Image Processing System

[![Python Version](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5.0%2B-green.svg)](https://opencv.org/)
[![Flask](https://img.shields.io/badge/Frontend-Flask%20%7C%20HTML5-orange.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![University](https://img.shields.io/badge/Institution-KR%20Mangalam%20University-red.svg)](https://www.krmangalam.edu.in/)

A comprehensive, modular, end-to-end computer vision and intelligent image processing framework implemented in Python with an interactive web-based user interface. This system encompasses the complete digital image processing pipeline: raw image acquisition, standardized preprocessing, synthetic noise modeling, image restoration, adaptive contrast enhancement, morphological segmentation, object feature extraction, quantitative evaluation metrics (MSE, PSNR, SSIM), and web dashboard visualization.

---

## Table of Contents
- [Problem Statement](#problem-statement)
- [Real-World Use Cases](#real-world-use-cases)
- [Tech Stack and Version Requirements](#tech-stack-and-version-requirements)
- [Web Application and Frontend Interface](#web-application-and-frontend-interface)
- [System Architecture and Workflow](#system-architecture-and-workflow)
- [Detailed Task Breakdown](#detailed-task-breakdown)
- [Quantitative Performance Evaluation](#quantitative-performance-evaluation)
- [Directory Structure](#directory-structure)
- [Quick Start and Installation](#quick-start-and-installation)
- [Usage Instructions](#usage-instructions)
- [Future Scope and Enhancements](#future-scope-and-enhancements)
- [Author and Academic Details](#author-and-academic-details)
- [License](#license)

---

## Problem Statement

In practical computer vision and digital image processing applications, optical sensors and image capture hardware frequently introduce degradation due to atmospheric distortion, inadequate illumination, sensor thermal noise, and electronic interference. Raw visual data cannot be reliably processed by downstream analytical or machine learning algorithms without prior systematic enhancement and restoration.

### Key Challenges Addressed
1. **Noise Degradation:** Mitigation of additive Gaussian thermal noise and impulsive Salt-and-Pepper transmission noise.
2. **Illumination Variations:** Rectification of non-uniform lighting conditions that impair global edge detection and static thresholding algorithms.
3. **Segmentation Accuracy:** Separation of foreground target objects from complex backgrounds using automatic and adaptive spatial thresholding combined with morphological operators.
4. **Feature Representation:** Extraction of scale- and rotation-invariant keypoints (ORB), structural contours, and edge boundaries for object recognition.
5. **Objective Quality Assessment:** Computation of mathematical and perceptual evaluation metrics (MSE, PSNR, SSIM) to quantitatively validate image restoration performance.

This project delivers an automated, 7-stage processing pipeline to address these challenges with reproducible statistical verification.

---

## Real-World Use Cases

| Domain / Industry | Application Scenario | Primary Technique / Module |
| :--- | :--- | :--- |
| **Medical Diagnostics** | Denoising MRI, CT, and X-ray images; segmenting tissue structures and lesion boundaries | CLAHE, Median Filter, Otsu Thresholding |
| **Industrial Inspection** | Detecting micro-cracks, surface defects, and automated component verification on assembly lines | Morphological Gradient, Canny Edges, Contour Analysis |
| **Autonomous Systems** | Road sign classification, lane detection, and feature point tracking under low visibility | ORB Descriptors, Adaptive Gaussian Thresholding |
| **Document Processing & OCR** | Restoring faded historical texts, removing background paper artifacts, and binarizing low-contrast scans | Adaptive Thresholding, CLAHE |
| **Microscopy & Research** | Quantitative cell counting, structural area measurement, and image degradation benchmarking | Bounding Boxes, SSIM / PSNR Metrics |

---

## Tech Stack and Version Requirements

### Backend & Core Algorithms
- **Python:** Version `3.7+` (Tested and recommended: `Python 3.8` through `3.11`)

### Frontend & Web Application
- **Web Framework:** Flask (`v2.0+`)
- **UI Architecture:** HTML5, CSS3, JavaScript, Jinja2 Template Engine (`templates/`)

### Core Libraries and Dependencies

| Library / Package | Minimum Required Version | Functional Purpose |
| :--- | :--- | :--- |
| **`opencv-python`** | `>= 4.5.0` | Digital image matrix transformations, color space conversions, spatial filtering, morphological operations, ORB feature extraction |
| **`flask`** | `>= 2.0.0` | Light-weight Web server and routes for web dashboard rendering (`app.py`) |
| **`numpy`** | `>= 1.21.0` | Multi-dimensional array operations, matrix mathematics, synthetic noise generation routines |
| **`matplotlib`** | `>= 3.4.0` | Rendering multi-panel pipeline comparison plots, intensity histograms, and saved visual outputs |
| **`scipy`** | `>= 1.7.0` | Advanced scientific computing functions and spatial signal filtering kernels |
| **`scikit-image`** | `>= 0.18.0` | Structural Similarity Index Measure (SSIM) and perceptual image quality algorithms |
| **`scikit-learn`** | `>= 0.24.0` | Data analytics tools and evaluation metric calculations |
| **`Pillow` (PIL)** | `>= 8.3.0` | Image file input/output validation and file format support |

---

## Web Application and Frontend Interface

The system features an interactive, browser-based web dashboard (`app.py`) built with Flask and responsive HTML5/CSS3 templates (`templates/`). This interface enables users to interact with the underlying computer vision algorithms without command-line invocation.

### Key Frontend Features
1. **Interactive Image Upload:** Drag-and-drop or upload custom image files (JPEG, PNG, WebP, BMP, TIFF).
2. **Real-Time Visual Stage Comparisons:** Side-by-side display comparing Original, Degraded, Denoised, Enhanced, Segmented, and Feature-Extracted visual outputs.
3. **Dynamic Parameter Control:** Tune filter kernel sizes, thresholding values, noise variance ($\sigma$), and CLAHE parameters through the UI.
4. **Live Metric Reporting:** Real-time computation and display of MSE, PSNR, and SSIM values directly on the web page.

### Launching the Web Interface
```bash
python app.py
```
Navigate to `http://localhost:5000` or `http://127.0.0.1:5000` in your web browser.

---

## System Architecture and Workflow

```text
┌────────────────────────────────────────────────────────────────────────┐
│               Interactive Web UI (Flask / HTML5 Templates)             │
│                      http://localhost:5000 (app.py)                    │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ Uploads / Triggers Pipeline
                                   ▼
┌─────────────────┐     ┌───────────────────┐     ┌────────────────────┐
│   Raw Image     │ ──> │  Task 2: Load &   │ ──> │ Task 3: Noise &    │
│  Acquisition    │     │  Preprocess       │     │ Restoration/CLAHE  │
└─────────────────┘     └───────────────────┘     └────────────────────┘
                                                            │
┌─────────────────┐     ┌───────────────────┐               ▼
│ Task 7: Full    │ <── │ Task 6: Metric    │ <── ┌────────────────────┐
│ Pipeline View   │     │ Evaluation        │     │ Task 4: Segment &  │
└─────────────────┘     └───────────────────┘     │ Morphological Ops  │
                                ▲                 └────────────────────┘
                                │                           │
                                └───────────────────────────┘
                                  Task 5: Features & ORB
```

---

## Detailed Task Breakdown

### Task 1: Project Setup and Architecture (`task1_setup.py`)
- Standardized directory layout for raw inputs, modular functional code, and output artifacts.
- Environment verification and logging module for execution tracking.

### Task 2: Image Acquisition and Preprocessing (`task2_preprocessing.py`)
- **Supported Formats:** JPEG, PNG, WebP, BMP, TIFF.
- **Dimensional Standardization:** Resizing arbitrary inputs to a uniform $512 \times 512$ resolution matrix.
- **Color Space Transformation:** Converting multi-channel RGB images to single-channel Grayscale matrices.

### Task 3: Image Enhancement and Restoration (`task3_enhancement.py`)
- **Synthetic Degradation Modeling:**
  - **Gaussian Noise:** Simulation of thermal sensor noise ($\mathcal{N}(\mu=0, \sigma=25)$).
  - **Salt-and-Pepper Noise:** Impulse transmission noise ($p = 0.05$).
- **Spatial Denoising Filters:**
  - **Mean Filter:** $5 \times 5$ linear spatial smoothing kernel.
  - **Median Filter:** $5 \times 5$ non-linear rank-order filter for impulsive noise elimination.
  - **Gaussian Filter:** Isotropic spatial smoothing kernel ($\sigma=1.5$).
- **Contrast Optimization:**
  - **Global Histogram Equalization:** Uniform redistribution of pixel intensity histograms.
  - **CLAHE:** Contrast Limited Adaptive Histogram Equalization ($\text{tile\_size}=8 \times 8, \text{clip\_limit}=3.0$).

### Task 4: Image Segmentation and Morphological Operations (`task4_segmentation.py`)
- **Binarization Methods:**
  - **Global Thresholding:** Static intensity cutoff ($T = 127$).
  - **Otsu's Thresholding:** Automated bimodal threshold selection via inter-class variance maximization.
  - **Adaptive Thresholding:** Localized Gaussian-weighted neighborhood thresholding.
- **Morphological Processing:**
  - **Dilation and Erosion:** Expansion and contraction of foreground structural elements.
  - **Opening:** Sequential erosion and dilation for isolated background noise removal.
  - **Closing:** Sequential dilation and erosion for hole filling and boundary bridging.
  - **Morphological Gradient:** Difference between dilated and eroded images to isolate structural boundaries.

### Task 5: Object Representation and Feature Extraction (`task5_features.py`)
- **Edge Extraction:**
  - **Sobel Operator:** First-order spatial gradient computation ($G_x, G_y$).
  - **Canny Detector:** Multi-stage edge detection featuring non-maximum suppression and hysteresis thresholding.
- **Contour Analysis:**
  - Identification of structural contours (3,174 identified) and bounding box generation around discrete objects.
- **Keypoint Detection:**
  - **ORB (Oriented FAST and Rotated BRIEF):** Extraction of 498 scale- and rotation-invariant feature descriptors.

### Task 6: Performance Evaluation (`task6_evaluation.py`)
Quantitative verification comparing degraded and restored images across three objective standards:
- **MSE (Mean Squared Error):** Arithmetic average of squared intensity differences.
- **PSNR (Peak Signal-to-Noise Ratio):** Logarithmic signal-to-noise power ratio measured in decibels (dB).
- **SSIM (Structural Similarity Index Measure):** Perceptual structural preservation metric ($0.0$ to $1.0$).

### Task 7: Consolidated Pipeline Visualization (`task7_visualization.py`)
- Generation of a multi-panel visual comparison illustrating the sequential transformation:
  `Original Input` ➔ `Degraded Image` ➔ `Denoised Image` ➔ `Contrast Enhanced` ➔ `Segmented Mask` ➔ `Feature Extraction`.

---

## Quantitative Performance Evaluation

Performance benchmarks recorded during empirical test execution:

| Evaluation Metric | Degraded Image | Restored Image (Median Filter) | Net Performance Variation |
| :--- | :---: | :---: | :---: |
| **Mean Squared Error (MSE)** ↓ | `1567.67` | `53.86` | **96.56% Error Reduction** |
| **Peak Signal-to-Noise Ratio (PSNR)** ↑ | `16.18 dB` | `30.82 dB` | **+14.64 dB Signal Improvement** |
| **Structural Similarity (SSIM)** ↑ | `0.088` | `0.732` | **+0.644 Structural Retention** |

---

## Directory Structure

```text
Assignment-5-Intelligent-Image-Processing/
├── main.py                    # CLI execution entry point (Runs command line pipeline)
├── app.py                     # Web application server (Flask web interface)
├── task1_setup.py             # System verification module
├── task2_preprocessing.py     # Image acquisition and normalization
├── task3_enhancement.py       # Denoising filters and CLAHE implementation
├── task4_segmentation.py      # Thresholding and morphological operators
├── task5_features.py          # Edge detection, contours, and ORB descriptors
├── task6_evaluation.py        # Quantitative metric calculations (MSE, PSNR, SSIM)
├── task7_visualization.py     # Consolidated multi-panel visualization generator
├── utils.py                   # Shared utility functions
├── requirements.txt           # Dependency manifest
├── LICENSE                    # MIT License specification
├── README.md                  # System documentation
├── templates/                 # Frontend HTML5/CSS3 templates for web UI
│   └── index.html             # Web dashboard user interface
└── outputs/                   # Generated output artifacts
    ├── task2_preprocessing.png
    ├── task3_enhancement.png
    ├── task4_segmentation.png
    ├── task5_features.png
    └── task7_pipeline_visualization.png
```

---

## Quick Start and Installation

### Prerequisites
Verify that Python `3.7` or a newer version is installed on the system:
```bash
python --version
```

### Step 1: Clone Repository
```bash
git clone https://github.com/itsashish1/Assignment-5-Intelligent-Image-Processing.git
cd Assignment-5-Intelligent-Image-Processing
```

### Step 2: Configure Virtual Environment (Recommended)
- **Linux / macOS:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
- **Windows (Command Prompt / PowerShell):**
  ```cmd
  python -m venv venv
  venv\Scripts\activate
  ```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## Usage Instructions

### Option 1: Running the Interactive Web Application (Frontend UI)
To launch the browser-based Web Interface:
```bash
python app.py
```
Open your web browser and navigate to `http://localhost:5000`. You can upload custom images and view real-time processing results interactively.

### Option 2: Running the Command-Line Pipeline
To run all 7 tasks sequentially in batch mode and save visual outputs to disk:
```bash
python main.py
```
Output visualizations will be stored directly inside the `outputs/` directory.

### Option 3: Modular Programmatic Usage
Individual modules can be imported directly into custom Python scripts:
```python
import task2_preprocessing as task2
import task3_enhancement as task3

# Load image and convert to standardized 512x512 grayscale matrix
original_img, grayscale_img = task2.process_image("path/to/image.jpg")

# Apply noise simulation, median filter restoration, and CLAHE enhancement
noisy_img, restored_img, enhanced_img = task3.enhance_and_restore(grayscale_img)
```

---

## Future Scope and Enhancements

- **Hardware Acceleration:** Integration of CUDA and `CuPy` bindings for real-time video stream processing ($>60 \text{ FPS}$).
- **Deep Learning Architectures:** Incorporation of Convolutional Neural Networks (DnCNN) for deep denoising and UNet models for semantic segmentation.
- **Microservice Infrastructure:** Containerization via Docker and deployment as a RESTful API service using FastAPI.
- **Frontend Dashboard Enhancements:** Expanding the web UI with React.js / Vue.js for enhanced canvas image manipulation and real-time histogram plotting.

---

## Author and Academic Details

- **Author:** Ashish Yadav
- **Roll Number:** 2301010413
- **Degree Program:** B.Tech Computer Science & Engineering (CSE)
- **Institution:** K.R. Mangalam University
- **Course Assignment:** Assignment 5 - Intelligent Image Processing System
- **Repository URL:** [itsashish1/Assignment-5-Intelligent-Image-Processing](https://github.com/itsashish1/Assignment-5-Intelligent-Image-Processing)

---

## License

This project is released under the **MIT License**. Refer to the [LICENSE](LICENSE) file for complete details.
