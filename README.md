# TReLU-Mixer
**Enhancing MLP-Mixer with Trainable ReLU Activation Functions**

This repository contains the official implementation of the paper:

> **The Impact of Trainable ReLU on the Performance of MLP-Mixer**  
> Shiping Ye, Vladimir Golovko*, Hanna Khatskevich, Marta Chodyka, Piotr Lichograj, Kaiheng Dai  

## Abstract
We introduce a novel approach to incorporating **Trainable ReLU (TReLU)** activation functions into the **MLP-Mixer** architecture. Our contributions are threefold:
1.  **Modified Layer Normalization (MLN)**: Integrating TReLU into the normalization step to improve generalization.
2.  **Replacement of GELU**: Substituting standard GELU activations with TReLU in mixer layers for enhanced performance.
3.  **Architectural Efficiency**: Presenting a significantly simplified MLP-Mixer with 18x fewer parameters than the baseline, while achieving superior accuracy.

Our model achieves state-of-the-art results on CIFAR-10 and CIFAR-100, outperforming larger contemporary models like Dmixnet.

## Key Features
- ✅ **Modified Layer Normalization (MLN)** with integrated TReLU.
- ✅ **TReLU activation** in MLP mixer blocks (MLP₁ and MLP₂).
- ✅ **Lightweight design**: 1.1M parameters vs. 18M in baseline MLP-Mixer.
- ✅ **Improved accuracy**: Up to +3.45% on CIFAR-10 and +3.7% on CIFAR-100.
- ✅ **Dynamic L2 Regularization** for stable training of TReLU parameters.

## Environment Setup
This code is implemented in PyTorch.

```bash
# Create a new conda environment (optional)
conda create -n trelu-mixer python=3.9 -y
conda activate trelu-mixer

# Install PyTorch (Refer to https://pytorch.org/ for your CUDA version)
pip install torch==2.0.1 torchvision==0.15.2

# Install other dependencies
pip install numpy==1.22.1 einops pandas matplotlib tqdm
