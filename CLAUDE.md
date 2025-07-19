# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains DynVelocity model training code (v15) for spatial-temporal analysis of mouse heart MERFISH data. The project uses neural ODEs to model cell state transitions over time, combining gene expression and spatial coordinates.

## Core Architecture

### DynVelocity Model
- **Neural ODE Framework**: Uses `torchdiffeq` for continuous-time modeling of cell trajectories
- **Multi-modal Input**: Handles both gene expression features (30D) and 3D spatial coordinates
- **Spatial Attention**: Implements message passing between cells using `SpatialAttentionLayer`
- **Dual Velocity Prediction**: Separate heads for expression velocity and position velocity
- **Energy Regularization**: Optional kinetic energy term for trajectory smoothness

### Training Pipeline
- **Optimal Transport Losses**: Multiple methods (EMD, Sinkhorn, FGW) for comparing predicted and true cell distributions
- **Mini-batch Training**: Samples random subsets from each timepoint for scalability
- **Adaptive Step Sizes**: Coarse-to-fine ODE integration during training
- **Integrated Evaluation**: Real-time assessment using coverage metrics and Gromov-Wasserstein distances

## Key Commands

### Training
```bash
python train.py
```

### Configuration
All hyperparameters are centralized in the `CONFIG` dictionary in `train.py`:
- Model architecture: `input_dim`, `hidden_dim`, `position_dim`, `sigma`
- Training: `n_epochs`, `lr`, `mini_batch_size`, `expr_alpha`
- Loss method: `loss_method` (pot_emd, pot_sinkhorn, pot_fgw, geomloss)
- Evaluation: `eval_every`, `eval_timepoints`, `eval_coverage`, `eval_gw`

### Debug Mode
Set `debug_mode: True` in CONFIG to run only 1 epoch for testing.

## Data Structure

### Input Data Format
- **Data File**: `adata_list.v250715.pt` (1.5GB PyTorch tensor, excluded from git)
- **Structure**: List of AnnData objects, one per timepoint
- **Required Keys**:
  - `adata.obsm['Z_mean']`: Latent gene expression embeddings (30D)
  - `adata.obsm['std_3D']`: Standardized 3D spatial coordinates
  - `adata.obs['CombinedCellType']`: Cell type labels
  - `adata.obs['time']`: Timepoint information

### Dependencies
External module paths are hardcoded:
- `/scratch/users/chensj16/codes/dynode_training`
- `/scratch/users/chensj16/codes/dynode_development`

Requires: `dynamica.sat.SpatialAttentionLayer`, `dynamica.equi.E3NNVelocityPredictor`

## Training Output
- **Model Checkpoints**: Saved to `/scratch/users/chensj16/codes/dynode_training/mouse-data-tmp/`
- **Logs**: Written to `./logs/` directory
- **WandB Tracking**: Project `fgw-train250519` (configurable)

## Evaluation Metrics
- **Coverage**: Percentage of true cells within learned radius of predictions
- **Gromov-Wasserstein**: Spatial structure preservation metric
- **Label Consistency**: Cell type preservation through predicted trajectories
- **Velocity Norms**: Magnitude analysis of expression and position velocities