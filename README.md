# AER-SwinT
The source code for "Tumor Microenvironment-guided Deep Learning Predicts Chemotherapy and Immunotherapy Response in Gastric Cancer with Attention Enhanced Residual Swin-Transformer: A Multicentric Study"

<div align=center><img width="1200" height="380" src="figures/Framework%20main.png"/></div>
<p align="left"> 
The overview of our method. 
</p>



## Requirements
Python 3.10 is required. The script uses CUDA when available and otherwise runs on CPU. GPU acceleration requires an NVIDIA GPU with a CUDA 12.8-compatible driver. Package versions are pinned in `requirements.txt`.

## Data
In order to make it easier for the readers to reproduce and understand the code, I have provided a small amount of example data used in our experiment under the **dataset** folder, where provides six training, validation and test images.

## File declaration


**main.py**: The codes for training, validating and testing.

## Run the codes
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
