# AER-SwinT
The source code for "Tumor Microenvironment-guided Deep Learning Predicts Chemotherapy and Immunotherapy Response in Gastric Cancer with Attention Enhanced Residual Swin-Transformer: A Multicentric Study"

<div align=center><img width="1200" height="380" src="figures/Framework%20main.png"/></div>
<p align="left"> 
The overview of our method. 
</p>



## Requirements
Python 3.10 is required. The script uses CUDA when available and otherwise runs on CPU. GPU acceleration requires an NVIDIA GPU with a CUDA 12.8-compatible driver. Package versions are pinned in `requirements.txt`.

## Data
Four unique demo cases are provided under `image_samples/`. The same four cases are reused across the training, validation, and test splits (four case folders per split) to demonstrate workflow execution.

## File declaration


**main.py**: The codes for training, validating and testing.

## Run the codes
**Installation note for v1.1.0 (headless Linux servers):** In a fresh environment, remove `opencv-python==4.11.0.86` from `requirements.txt` and retain `opencv-python-headless==4.11.0.86` before running `pip install -r requirements.txt`. Both packages provide `cv2` and must not be installed together. The headless package avoids dependencies on graphical libraries.

Install the environment.
```bash
conda create -n aer-swint python=3.10 pip -y
conda activate aer-swint
pip install --upgrade pip
pip install -r requirements.txt
```

Train and test the model.
```bash
python main.py
```
