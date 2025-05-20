# B-VGGNet
B-VGGNet is a Bayesian deep learning framework designed for X-ray Diffraction identification.


Environment:

- **GPU**: 4× NVIDIA GeForce RTX 3090 (24GB)
- **CUDA**: 12.4 (compatible with PyTorch 2.5.1)
- **Framework**: PyTorch 2.5.1 (compiled with CUDA 12.4)
- **Python**: 3.9.13

You can follow demo.ipynb step-by-step to reproduce the full pipeline.

Dataset: 
- VSS and RSS datasets.

Data process:
- Data Augment and Data Synthesis

Model:
- B-VGGNet1.py (Monte Carlo Dropout)
- B-VGGNet2.py (Laplace Approximation)
- B-VGGNet3.py (Variational Inference)
