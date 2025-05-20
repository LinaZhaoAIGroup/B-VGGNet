# B-VGGNet
B-VGGNet is a Bayesian deep learning framework designed for X-ray Diffraction identification. It addresses key challenges in automated material classification. 
This project combines advanced Bayesian inference techniques with spectrum synthesis and physical data augmentation to improve robustness and trustworthiness in real-world scientific applications.

Environment Setup
Recommended Versions
Python: 3.9.13
PyTorch: 2.5.1 + CUDA 12.4


You can follow demo.ipynb step-by-step to reproduce the full pipeline.

Dataset: VSS and RSS datasets.

Data process:
Data Augment and Data Synthesis

Model:
B-VGGNet1.py (Monte Carlo Dropout)
B-VGGNet2.py (Laplace Approximation)
B-VGGNet3.py (Variational Inference)
