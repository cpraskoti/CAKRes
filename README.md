# CAKRes

Read our press release [here](https://github.com/cpraskoti/CAKRes/blob/main/prfaq/Press_Release.pdf). For Frequently Asked Questions, refer [here](https://github.com/cpraskoti/CAKRes/blob/main/prfaq/FAQ.pdf).

## Introduction
This is a project by **CAKRes Innovations** that aims to improve workflow of research scientists in natural sciences. 
It aims to do so by producing real-world accurate upscaled image of low-resolution simulation of physical phenomenon.

Currently, the project to focuses on Super Resolution in the domain of Fluid Dynamics, with domain expansion in the future.

## Motivation
This project was motivated by a necessity in one of our labs. We drew inspiration from [this project](https://github.com/erichson/SuperBench).

## Resources
We are training our models using the following resources:

- 1x RTX4090
- 1x RTX3090
- 64 GB Memory
- 5 TB Storage
- Apache Spark cluster

## Benchmark
We might use [this benchmark](https://arxiv.org/abs/2306.14070).

## Dataset

The dataset are available [here](https://drive.google.com/drive/folders/17CK5aiOUJVVLuuEH418aw_RKAD7t_821?usp=drive_link). Use your UT Email Adderss to access it.

## Creating virtual environment
### Create conda or any other virtual environenment
```conda create -n cakres python=3.11```
### Activate Environment
```conda activate cakres```

### Install dependencies
```pip install -r requirements.txt```

## Running experiments
### Running FNO Training

```bash
python FNO/fno_fluid.py \
    --data_path path/to/your/training/data.h5 \
    --val_data_path path/to/your/validation/data \
    --exp_name experiement_name \
    --scale 4 \
    --epochs 50 \
    --batch_size 4 \
    --lr 0.001
```

### Experiment Output

results will be saved in a directory structure like this:

```
experiments/
└── <exp_name>/              # Directory named from --exp_name argument
    ├── output.log           # Console output and logs
    ├── metrics.json         # Training/validation metrics
    ├── best_fno_model_s<scale>.pth   # Best model checkpoint 
    ├── fno_metrics_s<scale>.png    # Plot for training and validation loss
    └── fno_crop_viz_s<scale>_p<patch_size>_*.png # Visualization comparison images
```
