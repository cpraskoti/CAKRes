# NSKT Super Resolution with DeepONet

This notebook implements a neural network approach to super-resolution for fluid flow data using the DeepONet architecture with Fourier feature embeddings. It specifically targets the NSKT (Navier-Stokes-Kolmogorov-Turbulence) dataset.

## Overview

The implementation uses a Deep Operator Network (DeepONet) architecture that consists of:
- A Branch network (CNN) that processes low-resolution input
- A Trunk network (MLP with Fourier feature embedding) that processes coordinates
- Matrix multiplication of branch and trunk outputs to generate high-resolution predictions

The model includes physics-informed losses that enforce divergence-free constraints and vorticity consistency for fluid flow data.

## Dependencies

```bash
pip install torch torchvision h5py matplotlib numpy scipy scikit-image tqdm torchinfo
```

## Dataset

The code expects an HDF5 file with the following structure:
- Key: 'fields'
- Shape: (num_samples, 3, height, width)
- Channels: u-velocity, v-velocity, vorticity

Default path: "/mnt/d/Datasets/SuperBench_data/nskt_16k/train/nskt_Re16000.h5"

## Configuration

Key parameters:
- `SCALE_FACTOR`: Upsampling factor (default: 4)
- `HR_PATCH_SIZE`: High-resolution patch size (default: 256)
- `BATCH_SIZE`: Batch size for training (default: 128)
- `NUM_EPOCHS`: Training epochs (default: 200)
- `LEARNING_RATE`: Initial learning rate (default: 1e-3)
- `FOURIER_MAPPING_SIZE`: Size of Fourier feature embeddings (default: 256)

## Model Architecture

The DeepONet model consists of:

1. **Branch Network**:
   - Multiple convolutional layers to extract features from low-resolution input
   - Fully connected layers to map features to a latent space

2. **Trunk Network**:
   - Fourier feature embedding for coordinates
   - MLP to process embedded coordinates

3. **Feature Combination**:
   - Efficient matrix multiplication using `einsum` to combine branch and trunk features

## Physics-Informed Losses

The model incorporates three loss components:
1. Data fitting (MSE loss)
2. Divergence loss (for incompressible flow constraint)
3. Vorticity consistency loss

## How to Run

1. Set the data path to your HDF5 file location
2. Adjust hyperparameters if needed
3. Run all cells in the notebook

The training process:
1. Calculates normalization statistics
2. Splits data into training and testing sets
3. Trains the model with scheduled learning rate
4. Evaluates performance using PSNR and SSIM metrics
5. Visualizes results on test samples

## Results

The notebook produces:
- Training and validation loss curves
- PSNR and SSIM metrics over epochs
- Visualizations of low-resolution input, ground truth, and predicted high-resolution outputs
