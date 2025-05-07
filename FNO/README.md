# FNO for Fluid Flow Super-Resolution

This project implements Fourier Neural Operators (FNO) for super-resolving fluid flow data.

## Dependencies

*   Python 3.x
*   PyTorch
*   NumPy
*   h5py
*   Matplotlib
*   scikit-learn
*   basicsr (Required for `fno_fluid_cropped.py`)

## Install Dependencies
Install dependencies using pip:
```
pip install torch numpy h5py matplotlib scikit-learn basicsr
```

### Running `fno_fluid.py`

```bash
python FNO/fno_fluid.py \
    --data_path path/to/your/data.h5 \
    --exp_name my_full_image_experiment \
    --scale 4 \
    --epochs 50 \
    --batch_size 4 \
    --lr 0.001
```

## Experiment Output

results will be saved in a directory structure like this:

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
