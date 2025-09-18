
# Template Matching for Character Recognition

This repository demonstrates **Template Matching** techniques for detecting and recognizing characters from an alphabet grid image. It includes **Python (OpenCV)** ,**Python (Scratch)** and **MATLAB** implementations, along with ground truth annotations for evaluation.

---

## Repository Structure

```
├── alphabet_grid.png            # Source image containing all alphabets
├── templates/                   # Folder with individual character templates (Aa.png, Bb.png, ..., Zz.png)
│   ├── Aa.png
│   ├── Bb.png
│   ├── ...
│   └── Zz.png
├── ground_truth.txt             # File containing ground truth annotations
├── template_matching.ipynb      # Python (OpenCV) implementation in Jupyter Notebook
├── template_matching.m          # MATLAB implementation for single image
├── template_matching_multi.m    # MATLAB implementation for multiple images
└── README.md                    # Documentation (this file)
```

---

## Getting Started

### 1. Requirements

#### Python

* Python 3.x
* OpenCV
* NumPy
* Matplotlib

Install dependencies:

```bash
pip install opencv-python numpy matplotlib
```

#### MATLAB

* MATLAB R2020a or later (older versions may also work).
* Image Processing Toolbox (recommended).

---

### 2. Dataset

* **Source Image:** `alphabet_grid.png` – contains the full grid of English alphabets.
* **Templates:** Individual character images (`Aa.png`, `Bb.png`, etc.) stored in the `templates/` folder.
* **Ground Truth:** `ground_truth.txt` – contains expected positions of characters for evaluation.

---


## Python Implementation

 `template_matching.ipynb` includes two scripts for template matching:

1. **OpenCV-based implementation** – Demonstrates how to perform template matching using OpenCV’s `cv2.matchTemplate` function with the **TM\_CCOEFF\_NORMED** method. It locates the best match of a smaller image (the **template**) within a larger image (the **source**).

2. **NumPy-based implementation** – Recreates OpenCV’s `cv2.TM_CCOEFF_NORMED` method from scratch using only NumPy. It slides the template across the source image and computes the **normalized cross-correlation coefficient** at each location to measure similarity.

---


Output:
<ol>
  <li>Shows detected character with a red bounding box. 
<li>Prints correlation score and location.
</ol>


---

## MATLAB Implementation

* `template_matching.m` → Performs template matching on a **single image**.
* `template_matching_multi.m` → Runs template matching across **multiple test images**.

Both use MATLAB’s `normxcorr2` function for normalized cross-correlation.

---

## Evaluation

* Uses `ground_truth.txt` for accuracy measurement.
* Metrics: **Precision, Recall, Accuracy**.
* The evaluation compares predicted match locations with ground truth coordinates.


