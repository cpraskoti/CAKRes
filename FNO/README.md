# FNO for Fluid Flow Super-Resolution
This project implements Fourier Neural Operators (FNO) for super-resolving fluid flow data.

### Create Virtual environenment
```conda create -n cakres python=3.11```
### Activate Environment
```conda activate cakeres```

### Install dependencies
```pip install -r requirements.txt```

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
└── <exp_name>/              # Directory named from --exp_name argument
    ├── output.log           # Console output and logs
    ├── metrics.json         # Training/validation metrics
    ├── best_fno_model_s<scale>.pth   # Best model checkpoint 
    ├── fno_metrics_s<scale>.png    # Plot of training/validation metrics
    └── fno_crop_viz_s<scale>_p<patch_size>_*.png # Visualization comparison images
```
