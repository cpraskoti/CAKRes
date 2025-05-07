
import h5py
import os
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend BEFORE importing pyplot
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from timeit import default_timer
import operator
from functools import reduce
from functools import partial
import argparse
import json # For saving config
import csv  # For saving metrics
from datetime import datetime # For timestamping experiments
import logging

# Try to import cropping function (optional dependency if not cropping)
try:
    from basicsr.data.transforms import paired_random_crop
    from basicsr.utils.img_util import tensor2img # For visualization
    from sklearn.metrics import mean_squared_error, mean_absolute_error # If cropping
    HAS_BASICSCR = True
except ImportError:
    HAS_BASICSCR = False
    paired_random_crop = None
    tensor2img = None
    mean_squared_error = None
    mean_absolute_error = None

# --- Configuration ---
# Data parameters
# TODO: Adjust this path
DATA_PATH = "../CAKRes/DatasetRe16k/valid/nskt_Re16000-003.h5" # Or the path to your HDF5 file
SAMPLE_LIMIT = 100 # Use a subset of samples for faster testing, None for all
T_SKIP = 5         # Time downsampling factor
N_SKIP = 2         # Spatial downsampling factor for original data loading

# FNO Model parameters
SCALE = 4          # Super-resolution factor (LR grid -> HR grid)
MODES = 16         # Number of Fourier modes
WIDTH = 32         # Feature width in FNO layers - Reduced for memory. Start lower if OOM persists.

# Training parameters
BATCH_SIZE = 8    # Reduced for memory. Try 8 or 4 if OOM persists. Adjust LEARNING_RATE if needed.
LEARNING_RATE = 0.001
EPOCHS = 50        # Reduced for quicker testing
STEP_SIZE = 10     # Learning rate scheduler step size
GAMMA = 0.5        # Learning rate scheduler gamma
WEIGHT_DECAY = 1e-4

# Experiment Name
DEFAULT_EXP_NAME = 'fno_fluid_exp' # Default experiment name

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Loss function
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]
        h = 1.0 / (x.size()[1] - 1.0)
        all_norms = (h**(self.d/self.p)) * torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)
        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.where(y_norms == 0, torch.tensor(1e-8, device=y.device), y_norms)
        fraction = diff_norms / y_norms
        if self.reduction:
            if self.size_average:
                return torch.mean(fraction)
            else:
                return torch.sum(fraction)
        return fraction

    def __call__(self, x, y):
        return self.rel(x, y)

# Normalization
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()
        # Calculate mean and std across batch and spatial dimensions, keep channel dim
        self.mean = torch.mean(x, dim=(0, 1, 2), keepdim=True)
        self.std = torch.std(x, dim=(0, 1, 2), keepdim=True)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        # Consistently use overall stats for decoding in this context
        std = self.std + self.eps
        mean = self.mean
        x = (x * std) + mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()
        return self

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()
        return self

    def to(self, device): # Added .to() method
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

# Complex multiplication (updated for torch.fft)
def compl_mul2d_new(a, b):
    # a: (batch, in_channel, H, W) complex
    # b: (in_channel, out_channel, H, W) complex
    # returns: (batch, out_channel, H, W) complex
    op = partial(torch.einsum, "bixy,ioxy->boxy")
    return op(a, b)

# --- FNO Components (Updated for torch.fft) ---
class SpectralConv2d_new(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_new, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        # Weights are complex-valued
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def forward(self, x):
        # x: (batch, in_channel, H, W)
        batchsize = x.shape[0]
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft2(x, norm='ortho') # (batch, in_channel, H, W//2 + 1) complex

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            compl_mul2d_new(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            compl_mul2d_new(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm='ortho') # (batch, out_channel, H, W)
        return x

# --- FNO Model for Super-Resolution ---
class FNOFluidSR(nn.Module):
    def __init__(self, modes1, modes2, width, scale_factor, use_cropping=False, dropout_rate=0.1):
        super(FNOFluidSR, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.scale_factor = scale_factor
        self.use_cropping = use_cropping
        self.dropout_rate = dropout_rate # Store dropout rate

        # Input channels: 3 (u,v,w) + 2 (grid_x, grid_y) = 5
        self.fc0 = nn.Linear(5, self.width)

        # FNO layers operate on the low-resolution grid
        self.conv0 = SpectralConv2d_new(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_new(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_new(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_new(self.width, self.width, self.modes1, self.modes2)

        # Skip connections
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        # Batch Norm (optional, can sometimes help)
        self.bn0 = torch.nn.BatchNorm2d(self.width)
        self.bn1 = torch.nn.BatchNorm2d(self.width)
        self.bn2 = torch.nn.BatchNorm2d(self.width)
        self.bn3 = torch.nn.BatchNorm2d(self.width)

        # Upsampling layer
        if self.scale_factor > 1:
            self.upsample = nn.Upsample(scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        else:
            self.upsample = nn.Identity()

        # Final projection to output channels (u_hr, v_hr, w_hr)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 3) # Output 3 channels

        # Dropout layers (Consider making dropout_rate an argument)
        # Initialize dropout layer
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        # x: (batch, H_lr, W_lr, 5)  (u,v,w,x,y)
        batchsize = x.shape[0]
        size_x_lr, size_y_lr = x.shape[1], x.shape[2]

        # Lift to feature space
        x = self.fc0(x) # (batch, H_lr, W_lr, width)
        x = x.permute(0, 3, 1, 2) # (batch, width, H_lr, W_lr)

        # FNO Block 0
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = self.bn0(x1 + x2)
        x = F.gelu(x)
        x = self.dropout(x) # Dropout after activation

        # FNO Block 1
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = self.bn1(x1 + x2)
        x = F.gelu(x)
        x = self.dropout(x) # Dropout after activation

        # FNO Block 2
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = self.bn2(x1 + x2)
        x = F.gelu(x)
        x = self.dropout(x) # Dropout after activation

        # FNO Block 3
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = self.bn3(x1 + x2)
        # x = F.gelu(x) # No activation after last block
        # No dropout right after last FNO block before upsampling

        # Upsample to High Resolution grid
        x = self.upsample(x) # (batch, width, H_hr, W_hr)

        # Project to output space
        x = x.permute(0, 2, 3, 1) # (batch, H_hr, W_hr, width)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x) # Dropout after fc1 activation
        x = self.fc2(x) # (batch, H_hr, W_hr, 3) -> (u_hr, v_hr, w_hr)

        return x

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        return c

# --- Data Loading and Preprocessing ---
def load_fluid_data(filepath, t_skip, n_skip, sample_limit=None):
    """Prepares indices for lazy loading from HDF5 file.

    Args:
        filepath (str): Path to the HDF5 file.
        t_skip (int): Time downsampling factor.
        n_skip (int): Spatial downsampling factor (applied in Dataset.__getitem__).
        sample_limit (int, optional): Maximum number of time samples to consider. Defaults to None (all).

    Returns:
        tuple: (filepath, list_of_hdf5_indices, hr_shape_after_n_skip, n_skip) or (None, None, None, None) on error.
    """
    print(f"Preparing indices from {filepath}...")
    try:
        with h5py.File(filepath, 'r') as file: # Open temporarily to get shape and indices
            dataset = file['fields']
            original_shape = dataset.shape
            total_time_samples = original_shape[0]

            if sample_limit is None:
                sample_limit = total_time_samples
            else:
                sample_limit = min(sample_limit, total_time_samples)
            # Determine the actual HDF5 indices to load later
            hdf5_indices = list(range(0, total_time_samples, t_skip))
            if len(hdf5_indices) > sample_limit:
                 hdf5_indices = hdf5_indices[:sample_limit]

            num_selected_samples = len(hdf5_indices)
            if num_selected_samples == 0 and total_time_samples > 0:
                print("Warning: No samples selected with current t_skip and sample_limit.")
                # Decide if we should proceed with 0 samples or error out
                # return None, None, None, None # Option: Error out

            # Calculate the HR shape *after* n_skip will be applied
            hr_h = len(range(0, original_shape[2], n_skip))
            hr_w = len(range(0, original_shape[3], n_skip))
            hr_shape_after_n_skip = (hr_h, hr_w)

            print(f"Original data shape: {original_shape}")
            print(f"Selected {num_selected_samples} time indices (limit: {sample_limit}, t_skip: {t_skip}) for lazy loading.")
            print(f"Target HR spatial shape after n_skip={n_skip}: {hr_shape_after_n_skip}")

            return filepath, hdf5_indices, hr_shape_after_n_skip, n_skip

    except Exception as e:
        print(f"Error accessing or reading HDF5 file metadata: {e}")
        return None, None, None, None

def get_grid(shape):
    """Generates 2D grid coordinates normalized from 0 to 1.

    Args:
        shape: Tuple containing spatial dimensions (size_x, size_y).

    Returns:
        Tensor of shape (1, size_x, size_y, 2).
    """
    size_x, size_y = shape[-2], shape[-1] # Expects (H, W) or (B, H, W)

    gridx = torch.linspace(0, 1, size_x, dtype=torch.float)
    gridx = gridx.reshape(1, size_x, 1, 1).repeat(1, 1, size_y, 1) # Correct repeat for x-coords

    gridy = torch.linspace(0, 1, size_y, dtype=torch.float)
    gridy = gridy.reshape(1, 1, size_y, 1).repeat(1, size_x, 1, 1) # Correct repeat for y-coords

    return torch.cat((gridx, gridy), dim=-1) # Shape (1, size_x, size_y, 2)

class FluidFlowFNODataset(Dataset):
    def __init__(self, filepath, hdf5_indices, hr_shape, n_skip, scale_factor, use_cropping=False, patch_size=64):
        self.filepath = filepath
        self.hdf5_indices = hdf5_indices
        self.hr_shape = hr_shape # Shape AFTER n_skip is applied
        self.n_skip = n_skip
        self.scale_factor = scale_factor
        self.use_cropping = use_cropping
        self.patch_size = patch_size # This is HR patch size
        self.lr_patch_size = patch_size // scale_factor

        self.h5_file = None # Worker-specific file handle

        if self.use_cropping and not HAS_BASICSCR:
             raise ImportError("Basicsr library not found. Cannot use cropping. Install with 'pip install basicsr'")
        if self.use_cropping and (patch_size % scale_factor != 0):
            raise ValueError(f"Patch size ({patch_size}) must be divisible by scale_factor ({scale_factor}) for cropping.")

        self.num_samples = len(hdf5_indices)
        self.lr_shape = (self.hr_shape[0] // scale_factor, self.hr_shape[1] // scale_factor)

        # Precompute full LR grid if not cropping (needs device later)
        if not self.use_cropping:
            self.grid_lr_full = get_grid(self.lr_shape) # (1, H_lr, W_lr, 2)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Open HDF5 file if not already opened by this worker
        if self.h5_file is None:
            try:
                self.h5_file = h5py.File(self.filepath, 'r', swmr=True) # Enable SWMR for multiproc read?
            except Exception as e:
                 print(f"WORKER ERROR opening {self.filepath}: {e}")
                 # Handle error appropriately, maybe return dummy data or raise
                 raise e # Raising might be better to stop DataLoader

        # Get the actual HDF5 index for the requested sample index
        hdf5_idx = self.hdf5_indices[idx]

        try:
            # 1. Load HR data for the single sample from HDF5, applying n_skip
            # Shape: (3, H_orig, W_orig) -> slice -> (3, H_hr, W_hr)
            hr_data_raw = self.h5_file['fields'][hdf5_idx, :, ::self.n_skip, ::self.n_skip]
            # Convert to float32 and permute to (H_hr, W_hr, 3)
            hr_target_np = hr_data_raw.astype(np.float32).transpose(1, 2, 0)
        except Exception as e:
            print(f"WORKER ERROR reading index {idx} (HDF5 index {hdf5_idx}) from {self.filepath}: {e}")
            # Handle error: return dummy data or raise
            raise e

        # --- From here, logic is similar but operates on the loaded hr_target_np ---

        if self.use_cropping:
            # 2a. Create full LR image from HR numpy data (needed for cropping reference)
            hr_tensor_temp = torch.from_numpy(hr_target_np).permute(2, 0, 1).unsqueeze(0) # (1, 3, H_hr, W_hr)
            lr_permuted_full = F.interpolate(
                hr_tensor_temp,
                size=self.lr_shape,
                mode='bilinear',
                align_corners=False
            )
            lr_full_np = lr_permuted_full.squeeze(0).permute(1, 2, 0).numpy() # (H_lr, W_lr, 3)

            # 2b. Perform paired random crop
            hr_patch_np, lr_patch_np = paired_random_crop(
                hr_target_np, lr_full_np, self.patch_size, self.scale_factor
            )

            # 2c. Convert cropped patches to tensors
            hr_target = torch.from_numpy(hr_patch_np) # (patch_hr, patch_hr, 3)
            lr_data = torch.from_numpy(lr_patch_np)   # (patch_lr, patch_lr, 3)

            # 2d. Generate grid for the LR patch - Pass only spatial shape
            grid_lr_patch = get_grid(lr_data.shape[:2]).squeeze(0) # (patch_lr, patch_lr, 2)

            # 2e. Concatenate LR patch data and grid
            lr_input = torch.cat([lr_data, grid_lr_patch], dim=-1) # (patch_lr, patch_lr, 5)

        else: # Use full images
            hr_target = torch.from_numpy(hr_target_np) # (H_hr, W_hr, 3)

            # 2a. Create full LR input by downsampling HR tensor
            hr_permuted = hr_target.permute(2, 0, 1).unsqueeze(0) # (1, 3, H_hr, W_hr)
            lr_permuted = F.interpolate(
                hr_permuted,
                size=self.lr_shape,
                mode='bilinear',
                align_corners=False
            )
            lr_data = lr_permuted.squeeze(0).permute(1, 2, 0) # (H_lr, W_lr, 3)

            # 2b. Concatenate LR data with precomputed full LR grid coordinates
            grid_rep = self.grid_lr_full.squeeze(0) # (H_lr, W_lr, 2)
            lr_input = torch.cat([lr_data, grid_rep], dim=-1) # (H_lr, W_lr, 5)

        return {'x': lr_input, 'y': hr_target}

    # Optional: Add a __del__ method to try and close the file handle
    # Note: __del__ is not guaranteed to be called reliably, especially with workers.
    # def __del__(self):
    #     if self.h5_file is not None:
    #         try:
    #             self.h5_file.close()
    #             # print(f"Worker closed HDF5 file: {self.filepath}")
    #         except Exception as e:
    #             pass # Ignore errors during cleanup


# --- Main Training Script ---
def main(args):
    # === Experiment Setup ===
    exp_dir = os.path.join("experiments", args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Configure logging
    log_file = os.path.join(exp_dir, 'output.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler() # Also print to console
        ]
    )
    logger = logging.getLogger()
    logger.info("Starting FNO Fluid Flow Super-Resolution")
    logger.info(f"Experiment Directory: {exp_dir}")
    logger.info(f"Arguments: {args}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Using cropping: {args.use_cropping}")
    if args.use_cropping:
        logger.info(f"Patch size (HR): {args.patch_size}x{args.patch_size}")
        logger.info(f"Patch size (LR): {args.patch_size // args.scale}x{args.patch_size // args.scale}")

    # --- Data Loading ---
    logger.info(f"Loading training data from: {args.data_path}")
    # Load training data indices based on train parameters
    train_filepath, train_hdf5_indices, hr_shape_after_n_skip, n_skip_loaded = load_fluid_data(
        args.data_path, args.t_skip, args.n_skip, sample_limit=args.sample_limit
    )
    if train_filepath is None or len(train_hdf5_indices) == 0:
        logger.error("Failed to load training data indices or no training samples selected. Exiting.")
        exit()
    logger.info(f"Using {len(train_hdf5_indices)} indices for training.")

    # Check if a separate validation path is provided
    if args.val_data_path:
        logger.info(f"Loading validation data from separate file: {args.val_data_path}")
        # Load validation data using val_t_skip argument
        val_filepath, val_hdf5_indices, val_hr_shape, val_n_skip = load_fluid_data(
            args.val_data_path,
            t_skip=args.val_t_skip, # Use the new argument here
            n_skip=args.n_skip,
            sample_limit=None # Still using all samples from val file by default
        )
        if val_filepath is None or len(val_hdf5_indices) == 0:
            logger.error("Failed to load validation data indices or no validation samples found in the specified file. Exiting.")
            exit()

        # Basic check for compatibility (can be expanded)
        if hr_shape_after_n_skip != val_hr_shape:
            logger.warning(f"Training HR shape {hr_shape_after_n_skip} and Validation HR shape {val_hr_shape} differ after n_skip. Ensure model compatibility.")
        if n_skip_loaded != val_n_skip:
            # You might want to enforce they are the same or handle this differently
             logger.warning(f"Training n_skip {n_skip_loaded} and Validation n_skip {val_n_skip} differ. Using training n_skip ({n_skip_loaded}) for both datasets.")
             # Override val_n_skip for consistency in dataset creation if needed, or error out.

        logger.info(f"Using {len(val_hdf5_indices)} indices for validation.")

    else:
        # --- Original Splitting Logic (If no val_data_path) ---
        logger.info(f"No separate validation file provided. Splitting training data from {args.data_path}.")
        # Validation uses the same file and indices loaded initially for training must be split
        val_filepath = train_filepath
        all_hdf5_indices = train_hdf5_indices # Use the indices loaded (potentially limited by sample_limit)
        split_idx = int(0.8 * len(all_hdf5_indices))
        # Re-assign train_hdf5_indices to the first part of the split
        train_hdf5_indices = all_hdf5_indices[:split_idx]
        # Assign val_hdf5_indices to the second part
        val_hdf5_indices = all_hdf5_indices[split_idx:]

        logger.info(f"Total selected indices before split: {len(all_hdf5_indices)}")
        logger.info(f"Using {len(train_hdf5_indices)} indices for training after split.")
        logger.info(f"Using {len(val_hdf5_indices)} indices for validation after split.")
        if len(train_hdf5_indices) == 0 or len(val_hdf5_indices) == 0:
            logger.error("Error: Not enough data for train/validation split after index selection and potential limiting.")
            exit()

    # --- Dataset Creation ---
    # Uses train_filepath, train_hdf5_indices determined above
    train_dataset = FluidFlowFNODataset(train_filepath, train_hdf5_indices, hr_shape_after_n_skip, n_skip_loaded,
                                        scale_factor=args.scale,
                                        use_cropping=args.use_cropping,
                                        patch_size=args.patch_size)
    # Uses val_filepath, val_hdf5_indices determined above
    # Assuming same hr_shape and n_skip for validation dataset creation - adjust if needed
    val_dataset = FluidFlowFNODataset(val_filepath, val_hdf5_indices, hr_shape_after_n_skip, n_skip_loaded,
                                      scale_factor=args.scale,
                                      use_cropping=args.use_cropping, # Use same mode for validation
                                      patch_size=args.patch_size)

    # --- Normalizer Calculation (Always use Training Data) ---
    logger.info("Calculating normalizer statistics on a subset of training data...")
    # Consider reducing num_norm_samples if memory is extremely tight during startup.
    num_norm_samples = min(len(train_hdf5_indices), 100) # Base on final train indices length
    norm_indices_to_load = train_hdf5_indices[:num_norm_samples] # Use indices from training set
    norm_data_list = []
    try:
        # Ensure we load normalizer data from the training file path
        with h5py.File(train_filepath, 'r') as temp_file: # Use train_filepath
            for hdf5_idx in norm_indices_to_load:
                # Load HR data slice, apply n_skip, transpose
                hr_data_raw = temp_file['fields'][hdf5_idx, :, ::n_skip_loaded, ::n_skip_loaded]
                hr_target_np = hr_data_raw.astype(np.float32).transpose(1, 2, 0)
                norm_data_list.append(torch.from_numpy(hr_target_np))
    except Exception as e:
        logger.error(f"Failed to load data for normalizer calculation: {e}")
        exit()

    if not norm_data_list:
         logger.error("No data loaded for normalizer calculation. Exiting.")
         exit()

    y_normalizer_data = torch.stack(norm_data_list, dim=0) # Shape (N_norm, H_hr, W_hr, 3)
    logger.info(f"Normalizer stats calculated from {y_normalizer_data.shape[0]} samples.")

    y_normalizer = UnitGaussianNormalizer(y_normalizer_data)
    y_normalizer.to(device)

    # 5. Create DataLoaders
    # Set num_workers > 0 to leverage multiprocessing for data loading
    # Make sure persistent_workers=True if num_workers > 0 and PyTorch >= 1.8 for efficiency
    # If OOM errors persist, try reducing num_workers, even to 0.
    num_workers = 4 if device.type == 'cuda' else 0 # Often good to use workers with GPU
    persistent_workers = (num_workers > 0)
    logger.info(f"Using {num_workers} DataLoader workers.")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True,
                            persistent_workers=persistent_workers)
    # Use a separate, potentially smaller batch size for validation/evaluation
    logger.info(f"Using validation batch size: {args.eval_batch_size}")
    val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True,
                          persistent_workers=persistent_workers)

    # 6. Initialize Model
    model = FNOFluidSR(modes1=args.modes, modes2=args.modes, width=args.width, scale_factor=args.scale, use_cropping=args.use_cropping, dropout_rate=args.dropout_rate).to(device)
    logger.info(f"FNO Model Parameter Count: {model.count_params()}")
    logger.info(f"Using dropout rate: {args.dropout_rate}") # Log the rate

    # 7. Setup Optimizer, Scheduler, Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    myloss = LpLoss(size_average=False) # Use LpLoss for evaluation metric
    # Consider using Mixed Precision Training (torch.cuda.amp) for further memory savings
    # scaler = torch.cuda.amp.GradScaler() # Requires changes in training loop

    # 8. Training Loop
    logger.info(f"Starting training for {args.epochs} epochs...")
    best_val_loss = float('inf')
    metrics_history = [] # Store metrics per epoch
    epochs_no_improve = 0 # Counter for early stopping

    for epoch in range(args.epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0.0
        for batch in train_loader:
            x = batch['x'].to(device) # LR input + grid (N, H_lr, W_lr, 5)
            y = batch['y'].to(device) # HR target (N, H_hr, W_hr, 3)

            # Normalize HR target
            y_norm = y_normalizer.encode(y)

            optimizer.zero_grad()
            out_norm = model(x) # Model outputs normalized prediction
            # Add torch.cuda.amp.autocast() context manager here if using mixed precision
            # with torch.cuda.amp.autocast():
            #    out_norm = model(x)

            # Calculate loss on normalized data
            loss = F.mse_loss(out_norm.view(out_norm.size(0), -1), y_norm.view(y_norm.size(0), -1), reduction='mean')
            loss.backward() # Use scaler.scale(loss).backward() with mixed precision
            optimizer.step() # Use scaler.step(optimizer); scaler.update() with mixed precision
            # Consider adding torch.cuda.empty_cache() here if memory fragmentation is suspected (may slow down training)

            # Calculate L2 error on unnormalized data for tracking progress (optional)
            with torch.no_grad():
                 out = y_normalizer.decode(out_norm)
                 train_l2 += myloss(out.view(out.size(0), -1), y.view(y.size(0), -1)).item()


        scheduler.step()
        model.eval()
        val_l2 = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(device) # LR input + grid
                y = batch['y'].to(device) # HR target

                out_norm = model(x)
                out = y_normalizer.decode(out_norm) # Decode output for evaluation

                # Calculate relative L2 error on original scale
                val_l2 += myloss(out.view(out.size(0), -1), y.view(y.size(0), -1)).item()

        train_l2 /= len(train_dataset)
        val_l2 /= len(val_dataset)

        # Store metrics for this epoch
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_l2': train_l2,
            'val_l2': val_l2,
        }
        metrics_history.append(epoch_metrics)

        t2 = default_timer()
        logger.info(f'Epoch [{epoch+1}/{args.epochs}], Time: {t2-t1:.1f}s, Train L2: {train_l2:.4f}, Val L2: {val_l2:.4f}')

        # Save best model based on validation L2 loss & check for early stopping
        if val_l2 < best_val_loss - args.early_stopping_delta:
            best_val_loss = val_l2
            epochs_no_improve = 0 # Reset counter
            save_path = f'best_fno_fluid_model_s{args.scale}'
            if args.use_cropping:
                save_path += f'_crop{args.patch_size}'
            save_path += '.pth'
            full_save_path = os.path.join(exp_dir, save_path)
            torch.save(model.state_dict(), full_save_path)
            logger.info(f"Validation loss improved. Saved best model to {full_save_path} with Val L2: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            logger.info(f"Validation loss did not improve for {epochs_no_improve} epoch(s). Current best: {best_val_loss:.4f}")

        # Early stopping check
        if args.early_stopping_patience > 0 and epochs_no_improve >= args.early_stopping_patience:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs due to no improvement for {args.early_stopping_patience} consecutive epochs.")
            break # Exit the training loop

    # 9. Final Evaluation (Optional)
    logger.info("\nTraining finished. Loading best model...")
    model_filename = f'best_fno_fluid_model_s{args.scale}'
    if args.use_cropping:
        model_filename += f'_crop{args.patch_size}'
    model_filename += '.pth'
    load_path = os.path.join(exp_dir, model_filename)

    if os.path.exists(load_path):
        model.load_state_dict(torch.load(load_path))
        model.eval()
        final_val_l2 = 0.0
        # Clear cache before final evaluation
        logger.info("Clearing CUDA cache before final evaluation...")
        torch.cuda.empty_cache()
        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(device)
                y = batch['y'].to(device)
                out_norm = model(x)
                out = y_normalizer.decode(out_norm)
                final_val_l2 += myloss(out.view(out.size(0), -1), y.view(y.size(0), -1)).item()
        final_val_l2 /= len(val_dataset)
        logger.info(f"Final Best Model Validation L2 Loss: {final_val_l2:.4f}")
    else:
        logger.warning(f"Best model file not found at {load_path}. Skipping final evaluation.")

    # Save metrics history to JSON
    metrics_file_path = os.path.join(exp_dir, 'metrics.json')
    try:
        with open(metrics_file_path, 'w') as f:
            json.dump(metrics_history, f, indent=4)
        logger.info(f"Saved metrics history to {metrics_file_path}")
    except Exception as e:
        logger.error(f"Error saving metrics to {metrics_file_path}: {e}")

    # 10. Plot Loss Curves (Optional)
    # Extract data for plotting from metrics_history
    epochs_list = [m['epoch'] for m in metrics_history]
    train_losses = [m['train_l2'] for m in metrics_history]
    val_losses = [m['val_l2'] for m in metrics_history]

    loss_plot_filename = f'fno_fluid_loss_s{args.scale}'
    if args.use_cropping:
        loss_plot_filename += f'_crop{args.patch_size}'
    loss_plot_filename += '.png'
    loss_plot_path = os.path.join(exp_dir, loss_plot_filename)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train L2 Loss')
    plt.plot(val_losses, label='Validation L2 Loss')
    plt.title(f'FNO Training and Validation L2 Loss (Scale={args.scale}, Crop={args.use_cropping})')
    plt.xlabel('Epoch')
    plt.ylabel('Relative L2 Loss')
    plt.yscale('log') # Log scale often useful for loss plots
    plt.legend()
    plt.grid(True)
    plt.savefig(loss_plot_path)
    logger.info(f"Saved loss plot to {loss_plot_path}")
    plt.close() # Close plot to avoid display issues

    # 11. Visualize Results (Optional)
    logger.info("Visualizing results for a few validation samples (showing patches if cropping)...")
    model.eval()
    num_viz_samples = min(3, len(val_dataset)) # Visualize fewer if dataset is small
    # Clear cache before visualization
    logger.info("Clearing CUDA cache before visualization...")
    torch.cuda.empty_cache()
    if num_viz_samples == 0:
        logger.warning("No validation samples to visualize.")
    else:
        with torch.no_grad():
            batch_count = 0
            sample_count = 0
            while sample_count < num_viz_samples and batch_count < len(val_loader):
                try:
                    batch = next(iter(val_loader)) # Get a sample batch
                except StopIteration:
                    break # Should not happen if len(val_loader) > 0
                batch_count += 1

                x = batch['x'].to(device) # (batch, H_in, W_in, 5)
                y = batch['y'].to(device) # (batch, H_out, W_out, 3)

                # Ensure batch size doesn't exceed remaining samples needed
                current_batch_size = x.size(0)
                viz_in_batch = min(current_batch_size, num_viz_samples - sample_count)

                if viz_in_batch <= 0: continue # Should not happen, but safety check

                # Get predictions
                out_norm = model(x[:viz_in_batch])
                out = y_normalizer.decode(out_norm) # (viz_in_batch, H_out, W_out, 3)

                # Visualize each sample needed from this batch
                for i in range(viz_in_batch):
                    sample_idx_in_batch = i
                    global_sample_idx = sample_count + i

                    # Move to CPU and numpy for plotting
                    # Input needs slicing to get only u,v,w channels
                    x_np = x[sample_idx_in_batch, ..., :3].cpu().numpy() # (H_in, W_in, 3)
                    y_np = y[sample_idx_in_batch].cpu().numpy()          # (H_out, W_out, 3)
                    out_np = out[sample_idx_in_batch].cpu().numpy()      # (H_out, W_out, 3)

                    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                    component_idx = 0 # Visualize the 'u' component

                    # Determine common color range for this sample
                    vmin = min(x_np[..., component_idx].min(), y_np[..., component_idx].min(), out_np[..., component_idx].min())
                    vmax = max(x_np[..., component_idx].max(), y_np[..., component_idx].max(), out_np[..., component_idx].max())

                    # Plotting logic
                    title_suffix = " Patch" if args.use_cropping else ""

                    im = axes[0].imshow(x_np[..., component_idx], cmap='jet', vmin=vmin, vmax=vmax)
                    axes[0].set_title(f'Low Resolution Input{title_suffix} (u) - Sample {global_sample_idx}')
                    axes[0].set_xticks([]); axes[0].set_yticks([])

                    im = axes[1].imshow(out_np[..., component_idx], cmap='jet', vmin=vmin, vmax=vmax)
                    axes[1].set_title(f'FNO Super-Resolved Output{title_suffix} (u) - Sample {global_sample_idx}')
                    axes[1].set_xticks([]); axes[1].set_yticks([])

                    im = axes[2].imshow(y_np[..., component_idx], cmap='jet', vmin=vmin, vmax=vmax)
                    axes[2].set_title(f'High Resolution Target{title_suffix} (u) - Sample {global_sample_idx}')
                    axes[2].set_xticks([]); axes[2].set_yticks([])

                    plt.tight_layout()
                    viz_filename = f'fno_fluid_visualization_{global_sample_idx}_s{args.scale}'
                    if args.use_cropping: viz_filename += f'_crop{args.patch_size}'
                    viz_filename += '.png'
                    viz_path = os.path.join(exp_dir, viz_filename)
                    plt.savefig(viz_path)
                    logger.info(f"Saved visualization plot to {viz_path}")
                    plt.close(fig) # Close figure after saving

                sample_count += viz_in_batch # Update count of visualized samples

    logger.info("FNO Fluid Flow script completed.")

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="FNO for Fluid Flow Super-Resolution")

    # Experiment Args
    parser.add_argument('--exp_name', type=str, default=DEFAULT_EXP_NAME, help='Name for the experiment directory')

    # Data Args
    parser.add_argument('--data_path', type=str, default=DATA_PATH, help='Path to HDF5 data file for training')
    parser.add_argument('--val_data_path', type=str, default=None, help='Optional path to a separate HDF5 file for validation')
    parser.add_argument('--sample_limit', type=int, default=SAMPLE_LIMIT, help='Maximum number of samples to load from training data (None for all)')
    parser.add_argument('--t_skip', type=int, default=T_SKIP, help='Time downsampling factor during training data loading')
    parser.add_argument('--val_t_skip', type=int, default=T_SKIP, help='Time downsampling factor during validation data loading (defaults to training t_skip)')
    parser.add_argument('--n_skip', type=int, default=N_SKIP, help='Spatial downsampling factor during loading (applied to both train and val)')

    # Model Args
    parser.add_argument('--scale', type=int, default=SCALE, help='Super-resolution factor')
    parser.add_argument('--modes', type=int, default=MODES, help='Number of Fourier modes')
    parser.add_argument('--width', type=int, default=WIDTH, help='Feature width in FNO')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate for regularization')

    # Training Args
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=8, help='Batch size for validation and visualization (default: 1)')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, dest='learning_rate', help='Learning rate')
    parser.add_argument('--step_size', type=int, default=STEP_SIZE, help='Scheduler step size')
    parser.add_argument('--gamma', type=float, default=GAMMA, help='Scheduler gamma value')
    parser.add_argument('--weight_decay', type=float, default=WEIGHT_DECAY, help='Adam weight decay')

    # Cropping Args
    parser.add_argument('--use_cropping', action='store_true', help='Enable paired random cropping')
    parser.add_argument('--patch_size', type=int, default=64, help='HR patch size for cropping (LR patch size = patch_size // scale)')

    # Early Stopping Args
    parser.add_argument('--early_stopping_patience', type=int, default=0, help='Number of epochs to wait for improvement before stopping (0 or less to disable)')
    parser.add_argument('--early_stopping_delta', type=float, default=0.0, help='Minimum change in validation loss to qualify as improvement')

    args = parser.parse_args()

    # Handle None for sample_limit
    if args.sample_limit is not None and args.sample_limit <= 0:
        args.sample_limit = None

    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
