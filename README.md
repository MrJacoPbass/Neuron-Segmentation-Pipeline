# Automated Generation of Neuron Segmentation Masks from Fluorescence Microscopy

Source code for the manuscript:  
> **"Automated Generation of Neuron Segmentation Masks from Fluorescence Microscopy as a Foundation for Phase‐Contrast Deep Learning Applications"**

> **Note:** *The codebase is currently being actively refactored and documented for final publication.*

---

## Overview

This repository provides an automated, classical image processing pipeline (`_F2M_parallel.py`) to generate reliable binary and labeled neuron segmentation masks from fluorescence microscopy images without requiring pre-annotated deep learning training data.

---

## Installation & Setup

### Prerequisites
- **Python:** 3.12 (Recommended)
- **Hardware:** OpenCL/CUDA compatible GPU (required for `pyclesperanto_prototype` acceleration)

### Virtual Environment Setup
Because this pipeline depends on **NumPy 1.x** (`numpy==1.26.4`), running it inside an isolated virtual environment is strongly recommended to avoid breaking existing installations:

```bash
# 1. Clone this repository
git clone [https://github.com/MrJacoPbass/Neuron-Segmentation-Pipeline.git](https://github.com/MrJacoPbass/Neuron-Segmentation-Pipeline.git)
cd Neuron-Segmentation-Pipeline

# 2. Create and activate a virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install required dependencies
pip install -r requirements.txt
