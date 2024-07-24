# Multiscale State-Centric Planner

## Overview

This repository implements the Multiscale State-Centric Planner algorithm.

## Installation

Follow the instructions on https://github.com/seohongpark/HIQL to install the required packages:

```
conda create --name mscp python=3.8
conda activate mscp
pip install -r requirements.txt --no-deps
pip install "jax[cuda11_cudnn82]==0.4.3" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install CALVIN (optional)
# Download `calvin.gz` (dataset) following the instructions at https://github.com/clvrai/skimo and place it in the `data` directory.
cd calvin
./install.sh

# Install Procgen (optional)
# Download `level500.npz` (dataset) from https://drive.google.com/file/d/1l1yHwzCYxHdgnW55R5pyhKFzHFqIQiQC/view and place it in the `data/procgen` directory.
# Download `level1000.npz` (dataset) from https://drive.google.com/file/d/19MqYZUENWWP7dHzlZFKhdVnouSxqfl5A/view and place it in the `data/procgen` directory.
pip install procgen`
```

The code uses Weights & Biases for experiment logging. Therefore, please follow the instructions on https://docs.wandb.ai/quickstart to setup WandB first.

## Usages

```
# Pretrain
./scripts/pretrain_antmaze.sh
./scripts/pretrain_kitchen.sh
./scripts/pretrain_calvin.sh
./scripts/pretrain_procgen.sh

# After pretraining, set the run ID of pretraining in the find_pretrained_checkpoint function of afrl/algorithm/checkpoint.py.
# After that, the program should be able to find and download pretrained checkpoints for online RL.

# MSCP
./scripts/mscp_antmaze.sh
./scripts/mscp_kitchen.sh
./scripts/mscp_calvin.sh
./scripts/mscp_procgen.sh

# HIQL Baseline
./scripts/baseline_hiql_antmaze.sh
./scripts/baseline_hiql_kitchen.sh
./scripts/baseline_hiql_calvin.sh
./scripts/baseline_hiql_procgen.sh
```

