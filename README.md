# DynVelocity Training Scripts

Neural ODE-based training pipeline for modeling cellular dynamics in spatial-temporal single-cell data.

## Overview

This repository contains training code for the **DynVelocity model v15**, which uses neural ordinary differential equations (ODEs) to model continuous cell state transitions over time. The model combines gene expression dynamics with spatial movement, making it particularly suitable for spatial transcriptomics data like MERFISH.

## Model Architecture

### Core Components

1. **Neural ODE Framework**: Uses `torchdiffeq` for continuous-time modeling
2. **Multi-modal Input**: 
   - Gene expression features (30D latent embeddings)
   - 3D spatial coordinates
3. **Spatial Attention**: Message passing between neighboring cells
4. **Dual Velocity Prediction**:
   - Expression velocity head (spectral normalized MLPs)
   - Position velocity head (E3NN equivariant network)
5. **Energy Regularization**: Optional kinetic energy term for smooth trajectories

### Training Logic

1. **Data Loading**: Load AnnData objects with expression, spatial, and temporal information
2. **Mini-batch Sampling**: Random sampling from each timepoint for scalability
3. **ODE Integration**: Forward simulation from initial to target timepoints
4. **Loss Computation**: Optimal transport distance between predicted and true distributions
5. **Evaluation**: Coverage metrics and spatial structure preservation assessment

## Prerequisites

### Dependencies
```bash
# Core ML libraries
torch
torchdiffeq
numpy
pandas
matplotlib

# Configuration management
hydra-core
omegaconf

# Single-cell analysis
scanpy
anndata

# Optimal transport
ot  # POT library
geomloss  # Optional, for GeomLoss

# Spatial analysis
scipy
sklearn

# Logging and visualization
wandb
tqdm
```

**Installation:**
```bash
pip install torch torchdiffeq hydra-core omegaconf numpy pandas matplotlib
pip install scanpy anndata scipy scikit-learn wandb tqdm
pip install pot  # For optimal transport
pip install geomloss  # Optional, for additional OT methods
```

### Custom Modules
The code requires custom `dynamica` modules:
- `dynamica.sat.SpatialAttentionLayer`
- `dynamica.equi.E3NNVelocityPredictor`

Paths are currently hardcoded to:
- `/scratch/users/chensj16/codes/dynode_training`
- `/scratch/users/chensj16/codes/dynode_development`

### Hardware Requirements
- CUDA-compatible GPU (configured for `cuda:0`)
- Sufficient memory for large datasets (current data file is 1.5GB)

## Data Format

### Input Data Structure
- **File**: `adata_list.v250715.pt` (PyTorch tensor file)
- **Format**: List of AnnData objects, one per timepoint
- **Required keys**:
  - `adata.obsm['Z_mean']`: Gene expression embeddings (30D)
  - `adata.obsm['std_3D']`: Standardized 3D spatial coordinates
  - `adata.obs['CombinedCellType']`: Cell type annotations
  - `adata.obs['time']`: Timepoint information

## Usage

### Basic Training
```bash
python train.py
```

### Configuration with Hydra

Configuration is managed through **Hydra** with YAML files. Default settings are in `config.yaml`, and can be overridden via command line.

#### Using Default Configuration
```bash
python train.py
```

#### Override Single Parameters
```bash
# Modify learning rate
python train.py training.lr=1e-6

# Change epochs and batch size
python train.py training.n_epochs=1000 training.mini_batch_size=512

# Enable debug mode (1 epoch only)
python train.py debug_mode=true

# Switch device
python train.py device=cuda:1
```

#### Override Complex Parameters
```bash
# Modify evaluation timepoints
python train.py evaluation.eval_timepoints="[[1,2],[3,4]]"

# Change learning rate schedule
python train.py training.lr_schedule="{50:1e-5,100:5e-6}"

# Disable features
python train.py logging.use_wandb=false evaluation.enable_eval=false
```

#### Using Alternative Config Files
```bash
# Use comprehensive training plan (16 forward/reverse sequences)
python train.py --config-name=config_full

# Use custom config file
python train.py --config-name=my_config
```

#### Useful Combinations
```bash
# Quick debug run
python train.py debug_mode=true evaluation.enable_eval=false

# Reduced evaluation overhead
python train.py evaluation.eval_samples=1000 training.mini_batch_size=512

# High learning rate experiment
python train.py training.lr=1e-5 training.lr_schedule="{10:5e-6,50:1e-6}"

# Full bidirectional training with comprehensive sequences
python train.py --config-name=config_full training.n_epochs=1000
```

### Training Features
- **Learning Rate Scheduling**: Automatic LR adjustments at specified epochs
- **Gradient Clipping**: Prevents gradient explosion (starts at epoch 100)
- **Integrated Evaluation**: Real-time assessment every N epochs
- **WandB Logging**: Experiment tracking and visualization
- **Checkpointing**: Model saves every 20 epochs

## Evaluation Metrics

1. **Coverage**: Percentage of true cells within learned radius of predictions
2. **Gromov-Wasserstein**: Spatial structure preservation metric
3. **Label Consistency**: Cell type preservation through trajectories
4. **Velocity Norms**: Analysis of expression and position velocity magnitudes

## Output

- **Model Checkpoints**: Saved to `/scratch/users/chensj16/codes/dynode_training/mouse-data-tmp/`
- **Logs**: Written to `./logs/` directory
- **WandB Dashboard**: Project `fgw-train250519`

## Change History
- 2025-07-16: Initial commit with DynVelocity v15 training pipeline

## Future plans
[ ] add loss support for pot_partial_ot and pot_fugw
