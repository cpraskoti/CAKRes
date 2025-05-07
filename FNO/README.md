# FNO for Fluid Flow Super-Resolution

This project implements Fourier Neural Operators (FNO) for super-resolving fluid flow data.
It includes two main scripts:

1.  `FNO/fno_fluid_cropped.py`: Trains an FNO model using a patch-based approach inspired by SwinIR. Input and output are cropped patches of the flow fields.
2.  `FNO/fno_fluid.py`: (Assumed) Trains an FNO model on full-resolution flow fields.

## Features

*   Super-resolution of 2D fluid dynamics data (u, v, w components).
*   Implementation based on the Fourier Neural Operator architecture.
*   Patch-based training (`fno_fluid_cropped.py`) for potentially handling larger datasets and improving local detail.
*   Experiment logging: Saves results, metrics, model checkpoints, and visualizations to a dedicated directory.

## Dependencies

*   Python 3.x
*   PyTorch
*   NumPy
*   h5py
*   Matplotlib
*   scikit-learn
*   basicsr (Required for `fno_fluid_cropped.py`)

Install dependencies using pip:
```bash
pip install torch numpy h5py matplotlib scikit-learn basicsr
```

## Running Experiments

Both scripts now support an `--exp_name` argument to organize experiment outputs.

### Running `fno_fluid_cropped.py` (Patch-based)

To run the patch-based FNO training:

```bash
python FNO/fno_fluid_cropped.py \
    --data_path path/to/your/data.h5 \
    --exp_name my_cropped_experiment \
    --scale 4 \
    --patch_size 64 \
    --epochs 50 \
    --batch_size 16 \
    --lr 0.001
    # Add other arguments as needed (e.g., --modes, --width, --sample_limit)
```

### Running `fno_fluid.py` (Full Image - Assuming similar structure)

*(Note: This assumes `fno_fluid.py` has been similarly modified for experiment logging. If not, the `--exp_name` argument won't work until it is updated.)*

To run the full-image FNO training:

```bash
python FNO/fno_fluid.py \
    --data_path path/to/your/data.h5 \
    --exp_name my_full_image_experiment \
    --scale 4 \
    --epochs 50 \
    --batch_size 4 \
    --lr 0.001
    # Add other arguments as needed
```

## Experiment Output

After running an experiment, the results will be saved in a directory structure like this:

```
experiments/
└── <exp_name>/              # Directory named after your --exp_name argument
    ├── output.log           # Console output and logs
    ├── metrics.json         # Training/validation metrics per epoch (Loss, MAE, MSE)
    ├── best_fno_crop_model_s<scale>_p<patch_size>.pth  # Best model checkpoint (cropped script)
    ├── best_fno_model_s<scale>.pth   # Best model checkpoint (full image script - assumed name)
    ├── fno_crop_metrics_s<scale>_p<patch_size>.png # Plot of training/validation metrics (cropped)
    ├── fno_metrics_s<scale>.png    # Plot of training/validation metrics (full image - assumed name)
    └── fno_crop_viz_s<scale>_p<patch_size>_*.png # Visualization comparison images (cropped)
    └── fno_viz_s<scale>_*.png     # Visualization comparison images (full image - assumed name)
```

Replace `<exp_name>`, `<scale>`, and `<patch_size>` with the values used during the run.
