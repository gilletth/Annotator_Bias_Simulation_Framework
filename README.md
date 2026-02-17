# Annotator Bias Simulation Framework

This repository provides a simulation framework for modeling **annotator (observer) bias** in medical image segmentation tasks.  
The framework generates synthetic observer-specific segmentation masks from ground-truth labels by introducing controlled variability in **boundary delineation**, **edge uncertainty**, and **segmentation volume**, enabling systematic study of inter-observer effects in downstream machine learning models.


---

## Key Features

- **Observer-specific boundary perturbations**  
  B-spline deformation driven by directional fields aligned with the signed distance map (SDM) gradient and observer-specific center-of-mass (COM) shifts (taken from litertaure reported values).

- **Edge-aware uncertainty modeling**  
  Sobel-based edge detection is used to introduce contrast-based uncertainty (ie., higher variability when segmentation boundries have less contrast).

- **Volume bias simulation**  
  Signed distance maps are thresholded to match observer-specific volume ratio ranges (values again taken from the literature), modeling systematic over- and under-segmentation tendencies.
---

## Expected Data Structure

The framework expects a BraTS-style directory structure with one folder per case:
<ground_truth_dir>/
BraTS-MEN-XXXXX-000/
BraTS-MEN-XXXXX-000-t1c.nii.gz
BraTS-MEN-XXXXX-000-seg.nii.gz
BraTS-MEN-YYYYY-000/
BraTS-MEN-YYYYY-000-t1c.nii.gz
BraTS-MEN-YYYYY-000-seg.nii.gz
...


Segmentations are assumed to be binary masks (`seg > 0`).

---

## Installation

### Python dependencies

```bash
pip install numpy scipy SimpleITK

---
## Usage

### Basic execution

Run the simulation on a BraTS-style dataset:

```bash
python Run_Simulation.py \
  --ground-truth-dir /path/to/BraTS-MEN-Train \
  --output-dir /path/to/output_masks

Reproducible execution
To ensure deterministic results across runs, specify a random seed:
python Run_Simulation.py \
  --ground-truth-dir /path/to/BraTS-MEN-Train \
  --output-dir /path/to/output_masks \
  --seed 123


