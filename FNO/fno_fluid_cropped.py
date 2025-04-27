# -*- coding: utf-8 -*-
"""
fno_fluid_cropped.py

Applies Fourier Neural Operator (FNO) to fluid flow super-resolution
using a patch-based training approach similar to SwinIR.
"""

import h5py
import os
import numpy as np
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
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Try to import cropping function (Required for this script)
try:
    from basicsr.data.transforms import paired_random_crop
    from basicsr.utils.img_util import tensor2img # For visualization
    HAS_BASICSCR = True
except ImportError:
    print("ERROR: Basicsr library not found or import failed. This script requires basicsr.")
    print("Install with: pip install basicsr scikit-learn")
    exit()

# --- Default Configuration ---
DEFAULT_DATA_PATH = "../CAKRes/DatasetRe16k/valid/nskt_Re16000-003.h5"
DEFAULT_SAMPLE_LIMIT = 100
DEFAULT_T_SKIP = 5
DEFAULT_N_SKIP = 2
DEFAULT_SCALE = 4
DEFAULT_MODES = 16
DEFAULT_WIDTH = 64
DEFAULT_PATCH_SIZE = 64 # HR Patch size
DEFAULT_BATCH_SIZE = 16 # SwinIR used 16 for scale 4
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EPOCHS = 50
DEFAULT_STEP_SIZE = 10 # Matches previous FNO script
DEFAULT_GAMMA = 0.5  # Matches previous FNO script
DEFAULT_WEIGHT_DECAY = 1e-4

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# --- Utility Functions ---

def get_grid(shape):
    """Generates 2D grid coordinates normalized from 0 to 1 for a batch.

    Args:
        shape: Tuple containing batch size and spatial dimensions (batchsize, size_x, size_y).

    Returns:
        Tensor of shape (batchsize, size_x, size_y, 2).
    """
    batchsize, size_x, size_y = shape[0], shape[1], shape[2]
    gridx = torch.linspace(0, 1, size_x, dtype=torch.float)
    gridx = gridx.reshape(1, size_x, 1, 1).repeat(batchsize, 1, size_y, 1)
    gridy = torch.linspace(0, 1, size_y, dtype=torch.float)
    gridy = gridy.reshape(1, 1, size_y, 1).repeat(batchsize, size_x, 1, 1)
    return torch.cat((gridx, gridy), dim=-1) # Shape (batchsize, size_x, size_y, 2)


# Complex multiplication (updated for torch.fft)
def compl_mul2d_new(a, b):
    op = partial(torch.einsum, "bixy,ioxy->boxy")
    return op(a, b)

# --- FNO Components (Updated for torch.fft) ---
class SpectralConv2d_new(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_new, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x, norm='ortho')
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        modes1 = min(self.modes1, x_ft.shape[-2])
        modes2 = min(self.modes2, x_ft.shape[-1])
        out_ft[:, :, :modes1, :modes2] = compl_mul2d_new(x_ft[:, :, :modes1, :modes2], self.weights1[:, :, :modes1, :modes2])
        out_ft[:, :, -modes1:, :modes2] = compl_mul2d_new(x_ft[:, :, -modes1:, :modes2], self.weights2[:, :, :modes1, :modes2])
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm='ortho')
        return x

# --- FNO Model (Same as before, handles patch size) ---
class FNOFluidSR(nn.Module):
    def __init__(self, modes1, modes2, width, scale_factor):
        super(FNOFluidSR, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.scale_factor = scale_factor
        self.fc0 = nn.Linear(5, self.width) # u,v,w,x,y
        self.conv0 = SpectralConv2d_new(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_new(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_new(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_new(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm2d(self.width)
        self.bn1 = torch.nn.BatchNorm2d(self.width)
        self.bn2 = torch.nn.BatchNorm2d(self.width)
        self.bn3 = torch.nn.BatchNorm2d(self.width)
        if self.scale_factor > 1:
            self.upsample = nn.Upsample(scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        else:
            self.upsample = nn.Identity()
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 3) # Output u,v,w

    def forward(self, x_with_grid):
        # x_with_grid: (batch, H_in, W_in, 5) - Input MUST include grid coords
        x = self.fc0(x_with_grid) # (batch, H_in, W_in, width)
        x = x.permute(0, 3, 1, 2) # (batch, width, H_in, W_in)
        x1 = self.conv0(x); x2 = self.w0(x); x = self.bn0(x1 + x2); x = F.gelu(x)
        x1 = self.conv1(x); x2 = self.w1(x); x = self.bn1(x1 + x2); x = F.gelu(x)
        x1 = self.conv2(x); x2 = self.w2(x); x = self.bn2(x1 + x2); x = F.gelu(x)
        x1 = self.conv3(x); x2 = self.w3(x); x = self.bn3(x1 + x2)
        x = self.upsample(x) # (batch, width, H_out, W_out)
        x = x.permute(0, 2, 3, 1) # (batch, H_out, W_out, width)
        x = self.fc1(x); x = F.gelu(x)
        x = self.fc2(x) # (batch, H_out, W_out, 3) -> (u_hr, v_hr, w_hr)
        return x

    def count_params(self):
        c = 0; # ... (count params logic) ...
        for p in self.parameters(): c += reduce(operator.mul, list(p.size()))
        return c

# --- Data Loading ---
def load_fluid_data(filepath, t_skip, n_skip, sample_limit=None):
    """Loads fluid data similarly to swinir_fluid.py."""
    print(f"Loading data from {filepath}...")
    # ... (Data loading logic identical to previous fno_fluid.py) ...
    if not os.path.exists(filepath):
        print(f"Error: Data file not found at {filepath}")
        return None, None, None
    try:
        file = h5py.File(filepath, 'r'); dataset = file['fields']
    except Exception as e: print(f"Error HDF5: {e}"); return None, None, None
    total_samples = dataset.shape[0]
    if sample_limit is None: sample_limit = total_samples
    num_time_samples = min(int(sample_limit), int(total_samples / t_skip))
    if num_time_samples == 0 and total_samples > 0: num_time_samples = 1
    print(f"Original shape: {dataset.shape}, Loading {num_time_samples} samples (limit:{sample_limit}, t_skip:{t_skip}, n_skip:{n_skip})...")
    try:
        u = dataset[::t_skip, 0, ::n_skip, ::n_skip][:num_time_samples]
        v = dataset[::t_skip, 1, ::n_skip, ::n_skip][:num_time_samples]
        w = dataset[::t_skip, 2, ::n_skip, ::n_skip][:num_time_samples]
    except Exception as e: print(f"Error slicing: {e}"); file.close(); return None, None, None
    file.close()
    print(f"Loaded shapes: u:{u.shape}, v:{v.shape}, w:{w.shape}")
    if u.shape[0] == 0: print("Warning: 0 samples loaded."); return None, None, None
    u, v, w = u.astype(np.float32), v.astype(np.float32), w.astype(np.float32)
    return u, v, w

# --- Dataset for Cropped Data (SwinIR style) ---
class FluidFlowCroppedDataset(Dataset):
    def __init__(self, u, v, w, scale_factor, patch_size):
        assert u.shape == v.shape == w.shape, "U, V, W must have the same shape"
        self.scale_factor = scale_factor
        self.patch_size = patch_size # HR patch size
        self.lr_patch_size = patch_size // scale_factor

        if patch_size % scale_factor != 0:
            raise ValueError(f"Patch size ({patch_size}) must be divisible by scale_factor ({scale_factor}).")

        # Store HR data as numpy arrays for cropping
        self.u_hr_np = u
        self.v_hr_np = v
        self.w_hr_np = w
        self.num_samples = u.shape[0]
        self.hr_shape = u.shape[1:] # H, W
        self.lr_shape = (self.hr_shape[0] // scale_factor, self.hr_shape[1] // scale_factor)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Get HR data for the index
        hr_u = self.u_hr_np[idx]
        hr_v = self.v_hr_np[idx]
        hr_w = self.w_hr_np[idx]
        hr_target_np = np.stack([hr_u, hr_v, hr_w], axis=-1) # (H_hr, W_hr, 3)

        # Create full LR image for cropping reference
        hr_tensor_temp = torch.from_numpy(hr_target_np).permute(2, 0, 1).unsqueeze(0)
        lr_permuted_full = F.interpolate(
            hr_tensor_temp, size=self.lr_shape, mode='bilinear', align_corners=False
        )
        lr_full_np = lr_permuted_full.squeeze(0).permute(1, 2, 0).numpy()

        # Perform paired random crop using basicsr function
        hr_patch_np, lr_patch_np = paired_random_crop(
            hr_target_np, lr_full_np, self.patch_size, self.scale_factor
        )

        # Convert cropped patches to tensors and permute to BCHW
        # Mimics img2tensor(..., float32=True)
        hr_tensor = torch.from_numpy(np.ascontiguousarray(hr_patch_np)).permute(2, 0, 1).float()
        lr_tensor = torch.from_numpy(np.ascontiguousarray(lr_patch_np)).permute(2, 0, 1).float()

        return {'lq': lr_tensor, 'gt': hr_tensor} # lq: (3, H_lr, W_lr), gt: (3, H_hr, W_hr)


# --- Visualization Function (SwinIR style) ---
def visualize_results(model, dataloader, device, num_samples=3, scale_factor=4, save_prefix="fno_crop_viz"):
    print("Visualizing results...")
    model.eval()
    count = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if count >= num_samples:
                break

            lr_batch = batch['lq'].to(device) # (B, 3, H_lr, W_lr)
            hr_batch = batch['gt'].to(device) # (B, 3, H_hr, W_hr)
            batch_size_viz = lr_batch.size(0)

            # --- Pre-process for FNO ---
            lr_permuted = lr_batch.permute(0, 2, 3, 1) # (B, H_lr, W_lr, 3)
            grid_lr = get_grid(lr_permuted.shape[:3]).to(device) # (B, H_lr, W_lr, 2)
            fno_input = torch.cat([lr_permuted, grid_lr], dim=-1) # (B, H_lr, W_lr, 5)
            # --- FNO Forward Pass ---
            sr_permuted = model(fno_input) # (B, H_hr, W_hr, 3)
            # --- Post-process FNO output ---
            sr_batch = sr_permuted.permute(0, 3, 1, 2) # (B, 3, H_hr, W_hr)

            # Visualize one sample from the batch
            lr_img = tensor2img(lr_batch[0].unsqueeze(0), rgb2bgr=False) # HWC format
            sr_img = tensor2img(sr_batch[0].unsqueeze(0), rgb2bgr=False)
            hr_img = tensor2img(hr_batch[0].unsqueeze(0), rgb2bgr=False)

            # Plot
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            component_idx = 0 # Visualize the 'u' component

            # Find common range for visualization
            vmin = min(lr_img[..., component_idx].min(), sr_img[..., component_idx].min(), hr_img[..., component_idx].min())
            vmax = max(lr_img[..., component_idx].max(), sr_img[..., component_idx].max(), hr_img[..., component_idx].max())

            axes[0].imshow(lr_img[..., component_idx], cmap='jet', vmin=vmin, vmax=vmax)
            axes[0].set_title('Low Resolution Patch (u)')
            axes[0].set_xticks([]); axes[0].set_yticks([])

            axes[1].imshow(sr_img[..., component_idx], cmap='jet', vmin=vmin, vmax=vmax)
            axes[1].set_title('Super Resolved Patch (u)')
            axes[1].set_xticks([]); axes[1].set_yticks([])

            axes[2].imshow(hr_img[..., component_idx], cmap='jet', vmin=vmin, vmax=vmax)
            axes[2].set_title('High Resolution Patch (u)')
            axes[2].set_xticks([]); axes[2].set_yticks([])

            plt.tight_layout()
            save_path = f"{save_prefix}_{i}_s{scale_factor}.png"
            plt.savefig(save_path)
            print(f"Saved visualization: {save_path}")
            plt.close(fig) # Close plot to avoid display issues in non-interactive envs
            count += 1


# --- Main Training Script ---
def main(args):
    print("Starting FNO Fluid Flow Super-Resolution (Cropped Mode)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Scale: {args.scale}, HR Patch Size: {args.patch_size}x{args.patch_size}")

    # 1. Load Data
    u_data, v_data, w_data = load_fluid_data(args.data_path, args.t_skip, args.n_skip, sample_limit=args.sample_limit)
    if u_data is None: exit()

    # 2. Split Data
    split_idx = int(0.8 * len(u_data))
    u_train, v_train, w_train = u_data[:split_idx], v_data[:split_idx], w_data[:split_idx]
    u_val, v_val, w_val = u_data[split_idx:], v_data[split_idx:], w_data[split_idx:]
    print(f"Train samples: {len(u_train)}, Validation samples: {len(u_val)}")
    if len(u_train) == 0 or len(u_val) == 0: print("Error: Not enough data."); exit()

    # 3. Create Datasets (Cropped Mode Only)
    train_dataset = FluidFlowCroppedDataset(u_train, v_train, w_train,
                                           scale_factor=args.scale,
                                           patch_size=args.patch_size)
    val_dataset = FluidFlowCroppedDataset(u_val, v_val, w_val,
                                         scale_factor=args.scale,
                                         patch_size=args.patch_size) # Use same settings for val

    # 4. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 5. Initialize Model
    model = FNOFluidSR(modes1=args.modes, modes2=args.modes, width=args.width, scale_factor=args.scale).to(device)
    print(f"FNO Model Parameter Count: {model.count_params()}")

    # 6. Setup Optimizer, Scheduler, Loss (SwinIR style)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    criterion = nn.L1Loss().to(device) # Use L1 Loss like SwinIR

    # 7. Training Loop (SwinIR style metrics)
    print(f"Starting training for {args.epochs} epochs...")
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    train_maes, val_maes = [], []
    train_mses, val_mses = [], []

    for epoch in range(args.epochs):
        model.train()
        t1 = default_timer()
        epoch_train_loss, epoch_train_mae, epoch_train_mse = 0.0, 0.0, 0.0
        num_train_batches = 0

        for batch in train_loader:
            num_train_batches += 1
            lr_batch = batch['lq'].to(device) # (B, 3, H_lr, W_lr)
            hr_batch = batch['gt'].to(device) # (B, 3, H_hr, W_hr)

            # --- Pre-process for FNO ---
            lr_permuted = lr_batch.permute(0, 2, 3, 1) # (B, H_lr, W_lr, 3)
            grid_lr = get_grid(lr_permuted.shape[:3]).to(device) # (B, H_lr, W_lr, 2)
            fno_input = torch.cat([lr_permuted, grid_lr], dim=-1) # (B, H_lr, W_lr, 5)

            # --- FNO Forward Pass ---
            optimizer.zero_grad()
            sr_permuted = model(fno_input) # (B, H_hr, W_hr, 3)

            # --- Post-process FNO output ---
            sr_batch = sr_permuted.permute(0, 3, 1, 2) # (B, 3, H_hr, W_hr)

            # Calculate loss & metrics (comparing BCHW tensors)
            loss = criterion(sr_batch, hr_batch)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            # Metrics calculation
            sr_np = sr_batch.detach().cpu().numpy()
            hr_np = hr_batch.detach().cpu().numpy()
            epoch_train_mae += mean_absolute_error(hr_np.ravel(), sr_np.ravel())
            epoch_train_mse += mean_squared_error(hr_np.ravel(), sr_np.ravel())

        scheduler.step()

        # Validation
        model.eval()
        epoch_val_loss, epoch_val_mae, epoch_val_mse = 0.0, 0.0, 0.0
        num_val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                num_val_batches += 1
                lr_batch = batch['lq'].to(device) # (B, 3, H_lr, W_lr)
                hr_batch = batch['gt'].to(device) # (B, 3, H_hr, W_hr)

                # --- Pre-process for FNO ---
                lr_permuted = lr_batch.permute(0, 2, 3, 1)
                grid_lr = get_grid(lr_permuted.shape[:3]).to(device)
                fno_input = torch.cat([lr_permuted, grid_lr], dim=-1)
                # --- FNO Forward Pass ---
                sr_permuted = model(fno_input)
                # --- Post-process FNO output ---
                sr_batch = sr_permuted.permute(0, 3, 1, 2)

                # Calculate loss & metrics
                loss = criterion(sr_batch, hr_batch)
                epoch_val_loss += loss.item()
                sr_np = sr_batch.cpu().numpy()
                hr_np = hr_batch.cpu().numpy()
                epoch_val_mae += mean_absolute_error(hr_np.ravel(), sr_np.ravel())
                epoch_val_mse += mean_squared_error(hr_np.ravel(), sr_np.ravel())

        # Calculate average metrics for the epoch
        avg_train_loss = epoch_train_loss / num_train_batches if num_train_batches > 0 else 0
        avg_train_mae = epoch_train_mae / num_train_batches if num_train_batches > 0 else 0
        avg_train_mse = epoch_train_mse / num_train_batches if num_train_batches > 0 else 0
        avg_val_loss = epoch_val_loss / num_val_batches if num_val_batches > 0 else 0
        avg_val_mae = epoch_val_mae / num_val_batches if num_val_batches > 0 else 0
        avg_val_mse = epoch_val_mse / num_val_batches if num_val_batches > 0 else 0

        train_losses.append(avg_train_loss); val_losses.append(avg_val_loss)
        train_maes.append(avg_train_mae); val_maes.append(avg_val_mae)
        train_mses.append(avg_train_mse); val_mses.append(avg_val_mse)

        t2 = default_timer()
        print(f'Epoch [{epoch+1}/{args.epochs}], Time: {t2-t1:.1f}s')
        print(f'  Train - Loss: {avg_train_loss:.4f}, MAE: {avg_train_mae:.4f}, MSE: {avg_train_mse:.4f}')
        print(f'  Val   - Loss: {avg_val_loss:.4f}, MAE: {avg_val_mae:.4f}, MSE: {avg_val_mse:.4f}')

        # Save best model based on validation L1 loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = f'best_fno_crop_model_s{args.scale}_p{args.patch_size}.pth'
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path} with Val Loss: {best_val_loss:.4f}")

    # 8. Final Evaluation & Plotting (SwinIR style)
    print("\nTraining finished.")
    load_path = f'best_fno_crop_model_s{args.scale}_p{args.patch_size}.pth'
    if os.path.exists(load_path):
         print(f"Best model saved at: {load_path}")
         print(f"Final Best Model Validation L1 Loss: {best_val_loss:.4f}") # Already have best loss
    else:
        print("Best model file not found. Cannot report final validation loss.")


    # Plot Loss Curves
    loss_plot_path = f'fno_crop_metrics_s{args.scale}_p{args.patch_size}.png'
    plt.figure(figsize=(18, 5))
    # L1 Loss Plot
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train L1 Loss')
    plt.plot(val_losses, label='Val L1 Loss')
    plt.title('L1 Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    # MAE Plot
    plt.subplot(1, 3, 2)
    plt.plot(train_maes, label='Train MAE')
    plt.plot(val_maes, label='Val MAE')
    plt.title('Mean Absolute Error'); plt.xlabel('Epoch'); plt.ylabel('MAE'); plt.legend(); plt.grid(True)
    # MSE Plot
    plt.subplot(1, 3, 3)
    plt.plot(train_mses, label='Train MSE')
    plt.plot(val_mses, label='Val MSE')
    plt.title('Mean Squared Error'); plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(loss_plot_path)
    print(f"Saved metrics plot to {loss_plot_path}")
    # plt.show()

    # 9. Visualize Results
    if os.path.exists(load_path):
        model.load_state_dict(torch.load(load_path)) # Load best model for viz
        visualize_results(model, val_loader, device, scale_factor=args.scale,
                          save_prefix=f"fno_crop_viz_s{args.scale}_p{args.patch_size}")
    else:
        print("Skipping visualization as best model file not found.")

    print("FNO Cropped Fluid Flow script completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FNO for Fluid Flow Super-Resolution (Cropped Patch Mode)")
    # Args identical to previous fno_fluid.py, except --use_cropping is removed
    parser.add_argument('--data_path', type=str, default=DEFAULT_DATA_PATH, help='Path to HDF5 data file')
    parser.add_argument('--sample_limit', type=int, default=DEFAULT_SAMPLE_LIMIT, help='Maximum number of samples to load (0 or None for all)')
    parser.add_argument('--t_skip', type=int, default=DEFAULT_T_SKIP, help='Time downsampling factor')
    parser.add_argument('--n_skip', type=int, default=DEFAULT_N_SKIP, help='Spatial downsampling factor')
    parser.add_argument('--scale', type=int, default=DEFAULT_SCALE, help='Super-resolution factor')
    parser.add_argument('--modes', type=int, default=DEFAULT_MODES, help='Number of Fourier modes')
    parser.add_argument('--width', type=int, default=DEFAULT_WIDTH, help='Feature width in FNO')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='Training batch size')
    parser.add_argument('--lr', type=float, default=DEFAULT_LEARNING_RATE, dest='learning_rate', help='Learning rate')
    parser.add_argument('--step_size', type=int, default=DEFAULT_STEP_SIZE, help='Scheduler step size')
    parser.add_argument('--gamma', type=float, default=DEFAULT_GAMMA, help='Scheduler gamma value')
    parser.add_argument('--weight_decay', type=float, default=DEFAULT_WEIGHT_DECAY, help='Adam weight decay')
    parser.add_argument('--patch_size', type=int, default=DEFAULT_PATCH_SIZE, help='HR patch size for cropping (LR patch size = patch_size // scale)')

    args = parser.parse_args()
    if args.sample_limit is not None and args.sample_limit <= 0: args.sample_limit = None
    main(args)