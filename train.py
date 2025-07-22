#!/usr/bin/env python3
"""
Training script with integrated evaluation for DynVelocity model v15.
Evaluates model every few epochs during training using Coverage + Gromov-Wasserstein.
"""

import os
import time
import datetime
import sys
from math import sqrt
from omegaconf import DictConfig
import hydra

import numpy as np
import anndata
import torch
import torch.nn as nn
import torch.nn.init as init
import wandb
from torchdiffeq import odeint_adjoint as odeint
from torch.nn.utils.parametrizations import spectral_norm
from sklearn.neighbors import NearestNeighbors
import ot  # For Gromov-Wasserstein

# Add paths
sys.path.append('/scratch/users/chensj16/codes/dynode_training')
sys.path.append('/scratch/users/chensj16/codes/dynode_development')

# Import custom modules after path setup
from dynamica.sat import SpatialAttentionLayer
from dynamica.equi import E3NNVelocityPredictor

# Import local utilities
from utils import setup_logging, get_gn, get_lr, DictAverageMeter

# ================================================================================================
# CONFIGURATION
# ================================================================================================

# Default configuration (will be overridden by Hydra)
CONFIG = {
    # Paths
    'save_path': '/scratch/users/chensj16/codes/dynode_training/mouse-data-tmp/',
    'data_path': '/oak/stanford/groups/xiaojie/chensj16/analysis-proj/dynode-training/mm-heart-merfish-v6/adata_list.v250715.pt',
    'log_path': './logs/train_v15.log',  # Path for log file
    
    # Data keys
    'latent_key': 'Z_mean',  # Key in adata.obsm for latent embeddings
    'position_key': 'std_3D',  # Key in adata.obsm for position coordinates
    'celltype_key': 'CombinedCellType',  # Key in adata.obs for cell type labels
    
    # Device
    'device': 'cuda:0',
    
    # Model architecture
    'input_dim': 30,
    'output_dim': 30,
    'hidden_dim': 512,
    'position_dim': 3,
    'sigma': 100.0,
    'static_pos': False,
    'message_passing': True,
    'expr_autonomous': False,
    'pos_autonomous': False,
    'energy_regularization': True,
    
    # Training
    'n_epochs': 5000,
    'lr': 1e-8,
    'weight_decay': 1e-4,
    'mini_batch_size': 1024,
    'expr_alpha': 0.75,
    'energy_lambda': 0,
    
    # Learning rate scheduling
    'lr_schedule': {10:1e-6, 20:5e-4, 1000: 5e-5},  # None = no scheduling, or dict like {100: 5e-4, 500: 1e-4, 1000: 5e-5}
    
    # Loss configuration
    'loss_method': 'pot_emd',
    'geomloss_blur': 1e-8,
    'geomloss_scaling': 0.5,
    'geomloss_reach': None,
    'sinkhorn_reg': 1e-1,
    'fgw_alpha': 0.75,
    'fgw_eps': 1e-6,
    'ot_max_iter' : None, # None or a finite number
    
    # Unbalanced Sinkhorn parameters
    'unbalanced_reg_m': 1.0,
    'unbalanced_reg_div': 'kl',
    
    # Training schedule
    'coarse_epochs': 50,
    'coarse_step_size': 0.050,
    'fine_step_size':   0.025,
    'ode_method': 'euler',
    
    # Logging and checkpointing
    'use_wandb': True,
    'wandb_project': 'fgw-train250519',
    'save_every': 20,
    'print_every': 10,
    'print_step_loss': True,  # Set to False to disable step-wise loss printing
    
    # Debug mode
    'debug_mode': False,  # Set to True to run only 1 epoch and exit (for debugging)
    
    # Evaluation settings
    'enable_eval': True,  # Set to False to completely disable evaluation during training
    'eval_every': 20,  # Evaluate every 10 epochs
    'eval_timepoints': [[2, 3]],
    'eval_samples': None,  # None = use all samples, or set number for downsampling
    'eval_integration_method': 'rk4',
    'eval_step_size': 0.01,
    'radius_neighbors': 20,  # Increased from 5 to get larger radius
    
    # Evaluation metrics control
    'eval_coverage': True,  # Always compute coverage
    'eval_gw': False,  # Set to False to disable GW computation
    'eval_label_consistency': True,  # Set to False to disable label consistency
    'max_gw_samples': None,  # None = use all samples, or set number for GW subsampling
    'label_knn_k': 5,  # k for kNN voting to assign labels to predictions
    
    # Training plans
    'train_plans': [[2,3]],
    
    # Gradient clipping
    'gradient_clipping': True,  # Enable gradient clipping
    'clip_grad_norm': 20.0,  # Gradient norm clipping threshold
    'clip_start_epoch': 100,  # Start clipping after this epoch (default 100)
}

def update_config_from_hydra(cfg: DictConfig) -> dict:
    """Update CONFIG dictionary with values from Hydra configuration."""
    global CONFIG
    
    # Create updated config from Hydra
    updated_config = {
        # Paths
        'save_path': cfg.paths.save_path,
        'data_path': cfg.paths.data_path,
        'log_path': cfg.paths.log_path,
        
        # Data keys
        'latent_key': cfg.data.latent_key,
        'position_key': cfg.data.position_key,
        'celltype_key': cfg.data.celltype_key,
        
        # Device
        'device': cfg.device,
        
        # Model architecture
        'input_dim': cfg.model.input_dim,
        'output_dim': cfg.model.output_dim,
        'hidden_dim': cfg.model.hidden_dim,
        'position_dim': cfg.model.position_dim,
        'sigma': cfg.model.sigma,
        'static_pos': cfg.model.static_pos,
        'message_passing': cfg.model.message_passing,
        'expr_autonomous': cfg.model.expr_autonomous,
        'pos_autonomous': cfg.model.pos_autonomous,
        'energy_regularization': cfg.model.energy_regularization,
        
        # Training
        'n_epochs': cfg.training.n_epochs,
        'lr': cfg.training.lr,
        'weight_decay': cfg.training.weight_decay,
        'mini_batch_size': cfg.training.mini_batch_size,
        'expr_alpha': cfg.training.expr_alpha,
        'energy_lambda': cfg.training.energy_lambda,
        
        # Learning rate scheduling
        'lr_schedule': dict(cfg.training.lr_schedule) if cfg.training.lr_schedule else None,
        
        # Loss configuration
        'loss_method': cfg.loss.method,
        'geomloss_blur': cfg.loss.geomloss_blur,
        'geomloss_scaling': cfg.loss.geomloss_scaling,
        'geomloss_reach': cfg.loss.geomloss_reach,
        'sinkhorn_reg': cfg.loss.sinkhorn_reg,
        'fgw_alpha': cfg.loss.fgw_alpha,
        'fgw_eps': cfg.loss.fgw_eps,
        'ot_max_iter': cfg.loss.ot_max_iter,
        
        # Unbalanced Sinkhorn parameters
        'unbalanced_reg_m': cfg.loss.unbalanced_reg_m,
        'unbalanced_reg_div': cfg.loss.unbalanced_reg_div,
        
        # Training schedule
        'coarse_epochs': cfg.schedule.coarse_epochs,
        'coarse_step_size': cfg.schedule.coarse_step_size,
        'fine_step_size': cfg.schedule.fine_step_size,
        'ode_method': cfg.schedule.ode_method,
        
        # Logging and checkpointing
        'use_wandb': cfg.logging.use_wandb,
        'wandb_project': cfg.logging.wandb_project,
        'save_every': cfg.logging.save_every,
        'print_every': cfg.logging.print_every,
        'print_step_loss': cfg.logging.print_step_loss,
        
        # Debug mode
        'debug_mode': cfg.debug_mode,
        
        # Evaluation settings
        'enable_eval': cfg.evaluation.enable_eval,
        'eval_every': cfg.evaluation.eval_every,
        'eval_timepoints': cfg.evaluation.eval_timepoints,
        'eval_samples': cfg.evaluation.eval_samples,
        'eval_integration_method': cfg.evaluation.eval_integration_method,
        'eval_step_size': cfg.evaluation.eval_step_size,
        'radius_neighbors': cfg.evaluation.radius_neighbors,
        
        # Evaluation metrics control
        'eval_coverage': cfg.evaluation.eval_coverage,
        'eval_gw': cfg.evaluation.eval_gw,
        'eval_label_consistency': cfg.evaluation.eval_label_consistency,
        'max_gw_samples': cfg.evaluation.max_gw_samples,
        'label_knn_k': cfg.evaluation.label_knn_k,
        
        # Training plans
        'train_plans': cfg.train_plans,
        
        # Gradient clipping
        'gradient_clipping': cfg.gradient_clipping.enabled,
        'clip_grad_norm': cfg.gradient_clipping.clip_grad_norm,
        'clip_start_epoch': cfg.gradient_clipping.clip_start_epoch,
    }
    
    # Update global CONFIG
    CONFIG.update(updated_config)
    return CONFIG


# ================================================================================================
# MODEL DEFINITION
# ================================================================================================

class DynVelocity(nn.Module):
    """Simplified dynamic velocity prediction model."""
    
    def _init_weights(self, m):
        """Initialize model weights."""
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            init.ones_(m.weight)
            init.zeros_(m.bias)
    
    def __init__(self, config):
        """Initialize model with configuration."""
        super().__init__()
        
        # Store config for reference
        self.config = config
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.position_dim = config['position_dim']
        self.sigma = config['sigma']
        self.message_passing = config['message_passing']
        self.static_pos = config['static_pos']
        self.energy_regularization = config['energy_regularization']
        self.expr_autonomous = config['expr_autonomous']
        self.pos_autonomous = config['pos_autonomous']

        # Base MLP for expression features
        self.mlp_base = nn.Sequential(
            spectral_norm(nn.Linear(self.input_dim + self.position_dim, self.hidden_dim)), 
            nn.LayerNorm(self.hidden_dim), nn.SiLU(), 
            spectral_norm(nn.Linear(self.hidden_dim, self.hidden_dim)), 
            nn.LayerNorm(self.hidden_dim), nn.SiLU(), 
            spectral_norm(nn.Linear(self.hidden_dim, self.hidden_dim)), 
            nn.LayerNorm(self.hidden_dim), nn.SiLU(), 
            spectral_norm(nn.Linear(self.hidden_dim, self.hidden_dim))
        )
        
        # Spatial attention layers (optional)
        if self.message_passing:
            self.spatial_base1 = nn.Sequential(
                SpatialAttentionLayer(
                    input_dim=self.input_dim, p_dim=self.position_dim,
                    hidden_dim=self.hidden_dim, output_dim=self.hidden_dim, 
                    residual=True, message_passing=True, sigma=self.sigma, use_softmax=False),
                spectral_norm(nn.Linear(self.hidden_dim, self.hidden_dim)), nn.LayerNorm(self.hidden_dim), nn.SiLU(),
                spectral_norm(nn.Linear(self.hidden_dim, self.hidden_dim)), nn.LayerNorm(self.hidden_dim), nn.SiLU(),
                spectral_norm(nn.Linear(self.hidden_dim, self.hidden_dim))
            )
            
            self.spatial_base2 = nn.Sequential(
                SpatialAttentionLayer(
                    input_dim=self.hidden_dim, p_dim=self.position_dim,
                    hidden_dim=self.hidden_dim, output_dim=self.hidden_dim, 
                    residual=False, message_passing=True, sigma=self.sigma, use_softmax=False),
                spectral_norm(nn.Linear(self.hidden_dim, self.hidden_dim)), nn.LayerNorm(self.hidden_dim), nn.SiLU(),
                spectral_norm(nn.Linear(self.hidden_dim, self.hidden_dim)), nn.LayerNorm(self.hidden_dim), nn.SiLU(),
                spectral_norm(nn.Linear(self.hidden_dim, self.hidden_dim))
            )

        # Expression velocity predictor
        expr_input_dim = self.hidden_dim * 2 + (0 if self.expr_autonomous else 1)
        self.expr_head = nn.Sequential(
            spectral_norm(nn.Linear(expr_input_dim, self.hidden_dim)), 
            nn.LayerNorm(self.hidden_dim), nn.SiLU(), 
            spectral_norm(nn.Linear(self.hidden_dim, config['output_dim']))
        )
        
        # Position velocity predictor
        if not self.static_pos:
            pos_input_dim = self.hidden_dim * 2 + (0 if self.pos_autonomous else 1)
            self.pos_head = E3NNVelocityPredictor(
                n_scalars=pos_input_dim, n_vec3d=1, 
                scalar_hidden=128, vec3d_hidden=128, n_vec3d_out=1
            )

        self.apply(self._init_weights)

    def forward(self, t, state_tensor, args=None):
        """Forward pass through the model."""
        # Parse state tensor - cleaner approach
        expr_dim = self.input_dim
        pos_dim = self.position_dim
        
        # Remove energy dimension if present
        if self.energy_regularization:
            features = state_tensor[:, :-1]  # Remove last energy dimension
        else:
            features = state_tensor
        
        # Split features
        expr_features = features[:, :expr_dim]
        pos_features = features[:, expr_dim:expr_dim + pos_dim]
        
        # Combine for input
        X = torch.cat([expr_features, pos_features], dim=1)

        # Compute base representation
        H0 = self.mlp_base(X)
        
        # Compute spatial representation (if enabled)
        if self.message_passing:
            H1 = self.spatial_base1(X)
            H1 = self.spatial_base2(torch.cat([H1, pos_features], dim=1))
        else:
            H1 = torch.zeros_like(H0)
        
        # Combine representations
        H = torch.cat([H0, H1], dim=1)

        # Predict expression velocities
        if self.expr_autonomous:
            feature_velo = self.expr_head(H)
        else:
            t_expanded = t.expand(H.size(0), 1)
            feature_velo = self.expr_head(torch.cat([H, t_expanded], dim=1))

        # Predict position velocities
        if self.static_pos:
            pos_velo = torch.zeros_like(pos_features)
        else:
            if self.pos_autonomous:
                pos_input = H
            else:
                t_expanded = t.expand(H.size(0), 1)
                pos_input = torch.cat([H, t_expanded], dim=1)
            
            pos_velo = self.pos_head.forward(
                self.pos_head.prepare_input(pos_input, pos_features.reshape(-1, 3))
            )

        # Combine velocities
        velocity = torch.cat([feature_velo, pos_velo], dim=1)
        
        # Add energy if regularization is enabled
        if self.energy_regularization:
            kinetic_energy = 0.5 * (feature_velo ** 2).mean(dim=1) + 0.5 * (pos_velo ** 2).mean(dim=1)
            velocity = torch.cat([velocity, kinetic_energy.unsqueeze(1)], dim=1)
            
        return velocity

# ================================================================================================
# DATA LOADING
# ================================================================================================

def load_data(config, logger):
    """Load and prepare training data."""
    logger.info(f"Loading data from {config['data_path']}")
    
    try:
        adata_list = torch.load(config['data_path'], weights_only=False)
        adata_merged = anndata.concat(adata_list, join='outer')
        
        # Time points
        T_list = [float(adata.obs['time'].iloc[0]) for adata in adata_list]
        
        # Features - using configurable keys
        Z_mean_list = [torch.from_numpy(adata.obsm[config['latent_key']]).to(torch.float32).to(config['device']) for adata in adata_list]
        P_list = [torch.from_numpy(adata.obsm[config['position_key']]).to(torch.float32).to(config['device']) for adata in adata_list]
        
        # Cell type labels - using configurable key
        ct_list = [adata.obs[config['celltype_key']].values for adata in adata_list]
        
        logger.info(f"Loaded {len(adata_list)} time points with {len(adata_merged)} total cells")
        
        return {
            'adata_list': adata_list,
            'adata_merged': adata_merged,
            'T_list': T_list,
            'Z_mean_list': Z_mean_list,
            'P_list': P_list,
            'ct_list': ct_list
        }
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

def precalculate_latent_radius(data, config, logger):
    """Pre-calculate static latent radius for evaluation using GPU acceleration."""
    logger.info("Pre-calculating static latent radius for evaluation (per timepoint)")
    
    timepoint_radii = []
    k = config['radius_neighbors']
    
    for timepoint_idx in range(len(data['Z_mean_list'])):
        # Get data for this timepoint
        z = data['Z_mean_list'][timepoint_idx]  # Already on GPU
        n_cells = z.size(0)
        
        logger.info(f"  Computing radius for t{timepoint_idx} ({n_cells} cells)")
        
        # Use sklearn NearestNeighbors for memory-efficient computation
        z_cpu = z.cpu().numpy()
        
        # sklearn's NearestNeighbors is memory-efficient for large datasets
        nn = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(z_cpu)
        distances, _ = nn.kneighbors(z_cpu)
        
        # Get k-th nearest neighbor distances (excluding self at index 0)
        kth_nn_distances = distances[:, -1]  # k-th neighbor (last one)
        
        # Calculate 90th percentile
        timepoint_radius = np.percentile(kth_nn_distances, 90)
        timepoint_radii.append(timepoint_radius)
        
        logger.info(f"    t{timepoint_idx} radius: {timepoint_radius:.4f}")
        
        # Clear variables
        del z_cpu, distances, kth_nn_distances
    
    # Average across all timepoints
    static_radius = np.mean(timepoint_radii)
    
    logger.info(f"Individual timepoint radii: {[f'{r:.4f}' for r in timepoint_radii]}")
    logger.info(f"Average static latent radius: {static_radius:.4f}")
    
    return static_radius

# ================================================================================================
# EVALUATION FUNCTIONS
# ================================================================================================

def evaluate_model(model, data, config, static_radius, logger):
    """Evaluate model on all configured timepoint pairs."""
    model.eval()
    
    eval_results = {}
    
    for start_idx, end_idx in config['eval_timepoints']:
        logger.info(f"Evaluating t{start_idx} → t{end_idx}")
        
        # Sample cells for evaluation (configurable)
        if config['eval_samples'] is None:
            # Use all samples
            n_samples = data['Z_mean_list'][start_idx].size(0)
            sample_indices = np.arange(n_samples)
        else:
            # Downsample
            n_samples = min(config['eval_samples'], data['Z_mean_list'][start_idx].size(0))
            sample_indices = np.random.choice(data['Z_mean_list'][start_idx].size(0), n_samples, replace=False)
        
        # Initial state
        z0 = data['Z_mean_list'][start_idx][sample_indices]
        p0 = data['P_list'][start_idx][sample_indices]
        state0 = torch.cat([z0, p0], dim=1)
        
        # Add energy dimension if needed
        if config['energy_regularization']:
            energy_init = torch.zeros(state0.size(0), 1, device=state0.device, dtype=state0.dtype)
            state0 = torch.cat([state0, energy_init], dim=1)
        
        # True target data
        z_true = data['Z_mean_list'][end_idx].cpu().numpy()
        p_true = data['P_list'][end_idx].cpu().numpy()
        
        # Cell type labels
        if config['eval_label_consistency']:
            ct_start = data['ct_list'][start_idx][sample_indices]
            ct_end = data['ct_list'][end_idx]
        
        # Time span
        t_span = torch.tensor([data['T_list'][start_idx], data['T_list'][end_idx]]).to(config['device'])
        
        # Predict trajectories
        with torch.no_grad():
            predicted_states = odeint(
                model, y0=state0, t=t_span,
                method=config['eval_integration_method'],
                options={"step_size": config['eval_step_size']}
            )[-1]  # Final state
        
        # Parse predictions
        if config['energy_regularization']:
            z_pred = predicted_states[:, :config['input_dim']].cpu().numpy()
            p_pred = predicted_states[:, config['input_dim']:-1].cpu().numpy()
        else:
            z_pred = predicted_states[:, :config['input_dim']].cpu().numpy()
            p_pred = predicted_states[:, config['input_dim']:].cpu().numpy()
        
        # Calculate velocities for velocity norm analysis
        t_eval = torch.tensor(data['T_list'][end_idx]).to(config['device'])
        state_for_velocity = torch.cat([
            torch.from_numpy(z_pred).to(config['device']),
            torch.from_numpy(p_pred).to(config['device'])
        ], dim=1)
        
        if config['energy_regularization']:
            energy_for_velocity = torch.zeros(state_for_velocity.size(0), 1, 
                                            device=state_for_velocity.device, 
                                            dtype=state_for_velocity.dtype)
            state_for_velocity = torch.cat([state_for_velocity, energy_for_velocity], dim=1)
        
        with torch.no_grad():
            velocities = model(t_eval, state_for_velocity)
        
        # Parse velocities
        if config['energy_regularization']:
            v_expr = velocities[:, :config['input_dim']].cpu().numpy()
            v_pos = velocities[:, config['input_dim']:-1].cpu().numpy()
        else:
            v_expr = velocities[:, :config['input_dim']].cpu().numpy()
            v_pos = velocities[:, config['input_dim']:].cpu().numpy()
        
        # Calculate metrics with timing
        metrics = {}
        timing_info = {}
        
        # 0. Velocity norms
        t_start = time.time()
        expr_velocity_norms = np.linalg.norm(v_expr, axis=1)
        pos_velocity_norms = np.linalg.norm(v_pos, axis=1)
        
        metrics['expr_velocity_norm_mean'] = np.mean(expr_velocity_norms)
        metrics['pos_velocity_norm_mean'] = np.mean(pos_velocity_norms)
        timing_info['velocity_norms'] = time.time() - t_start
        
        # 1. Coverage in latent space (always computed)
        if config['eval_coverage']:
            t_start = time.time()
            # Use GPU for distance computation - much faster!
            z_pred_torch = torch.from_numpy(z_pred).to(config['device'])
            z_true_torch = torch.from_numpy(z_true).to(config['device'])
            latent_distances = torch.cdist(z_pred_torch, z_true_torch)
            min_distances_to_pred = torch.min(latent_distances, dim=0)[0].cpu().numpy()
            coverage = np.mean(min_distances_to_pred < static_radius)
            
            # Debug info to understand why coverage is 0
            min_dist = np.min(min_distances_to_pred)
            max_dist = np.max(min_distances_to_pred)
            mean_dist = np.mean(min_distances_to_pred)
            logger.info(f"  Coverage debug: radius={static_radius:.4f}, min_dist={min_dist:.4f}, max_dist={max_dist:.4f}, mean_dist={mean_dist:.4f}")
            
            metrics['coverage'] = coverage
            timing_info['coverage'] = time.time() - t_start
        
        # 2. Gromov-Wasserstein distance for spatial structure (optional)
        if config['eval_gw']:
            t_start = time.time()
            # Subsample for computational efficiency (configurable)
            if config['max_gw_samples'] is not None and len(p_pred) > config['max_gw_samples']:
                gw_pred_idx = np.random.choice(len(p_pred), config['max_gw_samples'], replace=False)
                p_pred_gw = p_pred[gw_pred_idx]
            else:
                p_pred_gw = p_pred
            
            if config['max_gw_samples'] is not None and len(p_true) > config['max_gw_samples']:
                gw_true_idx = np.random.choice(len(p_true), config['max_gw_samples'], replace=False)
                p_true_gw = p_true[gw_true_idx]
            else:
                p_true_gw = p_true
            
            # Distance matrices
            C_pred = ot.dist(p_pred_gw, metric='euclidean')
            C_true = ot.dist(p_true_gw, metric='euclidean')
            
            # Uniform weights
            p_pred_weights = np.ones(len(p_pred_gw)) / len(p_pred_gw)
            p_true_weights = np.ones(len(p_true_gw)) / len(p_true_gw)
            
            # Compute Gromov-Wasserstein distance
            gw_distance = ot.gromov_wasserstein2(C_pred, C_true, p_pred_weights, p_true_weights)
            metrics['gw_distance'] = gw_distance
            timing_info['gw_distance'] = time.time() - t_start
        
        # 3. Label consistency (optional)
        if config['eval_label_consistency']:
            t_start = time.time()
            # GPU-accelerated kNN search (keep the fast part)
            z_pred_torch = torch.from_numpy(z_pred).to(config['device'])
            z_true_torch = torch.from_numpy(z_true).to(config['device'])
            
            # Calculate distance matrix and find k nearest neighbors
            distances = torch.cdist(z_pred_torch, z_true_torch)
            _, nn_indices = torch.topk(distances, k=config['label_knn_k'], dim=1, largest=False)
            nn_indices = nn_indices.cpu().numpy()
            
            # Assign labels to predictions using kNN voting
            predicted_labels = []
            for neighbors in nn_indices:
                neighbor_labels = ct_end[neighbors]
                # Vote among neighbors
                unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
                predicted_label = unique_labels[np.argmax(counts)]
                predicted_labels.append(predicted_label)
            
            predicted_labels = np.array(predicted_labels)
            
            # Calculate label consistency
            label_consistency = np.mean(ct_start == predicted_labels)
            metrics['label_consistency'] = label_consistency
            timing_info['label_consistency'] = time.time() - t_start
        
        # Print timing information
        total_eval_time = sum(timing_info.values())
        timing_str = "  Eval timing: "
        for metric, t in timing_info.items():
            timing_str += f"{metric}={t:.2f}s "
        timing_str += f"total={total_eval_time:.2f}s"
        logger.info(timing_str)
        
        eval_results[(start_idx, end_idx)] = metrics
    
    return eval_results

# ================================================================================================
# TRAINING FUNCTIONS
# ================================================================================================

def setup_loss_function(config, logger):
    """Setup loss function based on configuration."""
    if config['loss_method'] == 'geomloss':
        import geomloss
        geomloss_fn = geomloss.SamplesLoss(
            blur=config['geomloss_blur'],
            scaling=config['geomloss_scaling'],
            reach=config['geomloss_reach'],
            backend='tensorized',
            debias=True
        )
        logger.info("Using geomloss for optimal transport")
        return geomloss_fn
    else:
        logger.info(f"Using POT method: {config['loss_method']}")
        return None

def compute_ot_loss(pred_state, true_state, config, geomloss_fn=None):
    """Compute optimal transport loss based on configured method."""
    if config['loss_method'] == 'geomloss':
        return geomloss_fn(pred_state, true_state)
    
    import ot
    
    # Uniform weights
    a = torch.ones(pred_state.shape[0], device=pred_state.device) / pred_state.shape[0]
    b = torch.ones(true_state.shape[0], device=true_state.device) / true_state.shape[0]
    
    # Compute cost matrix on GPU
    M = torch.cdist(pred_state, true_state)
    
    if config['loss_method'] == 'pot_emd':
        if config['ot_max_iter'] is None:
            ot_loss = ot.emd2(a, b, M)
        else:
            ot_loss = ot.emd2(a, b, M, numItermax=config['ot_max_iter'])
            
    elif config['loss_method'] == 'pot_sinkhorn':
        if config['ot_max_iter'] is None:
            ot_loss = ot.sinkhorn2(a, b, M, reg=config['sinkhorn_reg'])
        else:
            ot_loss = ot.sinkhorn2(a, b, M, reg=config['sinkhorn_reg'], numItermax=config['ot_max_iter'])
            
    elif config['loss_method'] == 'pot_fgw':
        # Split for FGW - assume last 3 dims are spatial
        pred_expr = pred_state[:, :-3]
        pred_spatial = pred_state[:, -3:]
        true_expr = true_state[:, :-3]
        true_spatial = true_state[:, -3:]
        
        M = torch.cdist(pred_expr, true_expr)
        C1 = torch.cdist(pred_spatial, pred_spatial)
        C2 = torch.cdist(true_spatial, true_spatial)
        
        ot_loss = ot.gromov.fused_gromov_wasserstein2(
            M, C1, C2, a, b, 
            loss_fun='square_loss',
            alpha=config['fgw_alpha'],
            epsilon=config['fgw_eps']
        )
    elif config['loss_method'] == 'pot_unbalanced_sinkhorn':
        ot_loss = ot.unbalanced.sinkhorn_unbalanced2(
            a, b, M, 
            reg=config['sinkhorn_reg'],
            reg_m=config['unbalanced_reg_m'],
            div=config['unbalanced_reg_div']
        )
    else:
        raise ValueError(f"Unknown loss method: {config['loss_method']}")
    
    return ot_loss

def train_step(model, optimizer, data, config, loss_fn, logger, epoch):
    """Execute one training step."""
    model.train()
    total_losses = []
    ot_losses = []
    kenergies = []
    
    for inds in config['train_plans']:
        optimizer.zero_grad()
        
        try:
            # Prepare initial state
            i = inds[0]
            idx0 = np.random.randint(0, data['P_list'][i].size(0), size=config['mini_batch_size'])
            z0, p0 = data['Z_mean_list'][i][idx0, ...], data['P_list'][i][idx0, ...]
            
            # Combine initial state
            state0 = torch.cat([z0, p0], dim=1)
            if config['energy_regularization']:
                energy_init = torch.zeros(state0.size(0), 1, device=state0.device, dtype=state0.dtype)
                state0 = torch.cat([state0, energy_init], dim=1)
            
            t_sol = torch.tensor(data['T_list'])[inds].to(torch.float32).to(config['device'])
            
            # Prepare target values
            z1_list, p1_list = [], []
            for j in inds[1:]:
                idx1 = np.random.randint(0, data['P_list'][j].size(0), size=config['mini_batch_size'])
                z1, p1 = data['Z_mean_list'][j][idx1, ...], data['P_list'][j][idx1, ...]
                z1_list.append(z1)
                p1_list.append(p1)
            
            # ODE integration
            if epoch < config['coarse_epochs']:
                state_predictions = odeint(
                    model, y0=state0, t=t_sol,
                    method=config['ode_method'],
                    options={"step_size": config['coarse_step_size']}
                )[1:]
            else:
                state_predictions = odeint(
                    model, y0=state0, t=t_sol,
                    method=config['ode_method'],
                    options={"step_size": config['fine_step_size']}
                )[1:]
            
            # Compute loss
            total_loss = None
            alpha = config['expr_alpha']
            
            for j in range(state_predictions.size(0)):
                pred_state = state_predictions[j]
                
                # Parse prediction
                if config['energy_regularization']:
                    z1_pred = pred_state[:, :config['input_dim']]
                    p1_pred = pred_state[:, config['input_dim']:-1]
                    kenergy = torch.abs(torch.mean(pred_state[:, -1]))
                else:
                    z1_pred = pred_state[:, :config['input_dim']]
                    p1_pred = pred_state[:, config['input_dim']:]
                    kenergy = torch.tensor(0.0, device=pred_state.device)
                
                # Combine for loss computation
                state_pred_combined = torch.cat([sqrt(alpha) * z1_pred, sqrt(1 - alpha) * p1_pred], dim=1)
                state_true_combined = torch.cat([sqrt(alpha) * z1_list[j], sqrt(1 - alpha) * p1_list[j]], dim=1)
                
                # Compute OT loss
                ot_loss = compute_ot_loss(state_pred_combined, state_true_combined, config, geomloss_fn=loss_fn)
                total_loss = ot_loss + config['energy_lambda'] * kenergy if total_loss is None else total_loss + ot_loss + config['energy_lambda'] * kenergy
            
            if total_loss is None or np.isnan(total_loss.item()) or np.isinf(total_loss.item()):
                logger.warning("NaN or Inf found in loss, skipping step")
                continue
            
            total_loss.backward()
            
            # Apply gradient clipping if enabled and after specified epoch
            grad_norm_before = get_gn(model)
            if config['gradient_clipping'] and epoch >= config['clip_start_epoch']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad_norm'])
                grad_norm_after = get_gn(model)
                if grad_norm_before > config['clip_grad_norm']:
                    logger.debug(f"Gradient clipped: {grad_norm_before:.3e} -> {grad_norm_after:.3e}")
            else:
                grad_norm_after = grad_norm_before
            
            optimizer.step()
            
            # Store metrics
            total_losses.append(total_loss.item())
            ot_losses.append(ot_loss.item())
            kenergies.append(kenergy.item())
            
            # Print step-level progress (if enabled)
            if config['print_step_loss']:
                print(f"From {inds[0]} to {inds[-1]}, loss={ot_loss.item():.3e}, gn={grad_norm_after:.3e}")
            
        except Exception as e:
            logger.error(f"Error in training step: {e}")
            continue
    
    # Return metrics
    if len(total_losses) > 0:
        avg_total_loss = np.mean(total_losses)
        avg_ot_loss = np.mean(ot_losses)
        avg_kenergy = np.mean(kenergies)
        reg_ratio = config['energy_lambda'] * avg_kenergy / avg_ot_loss if avg_ot_loss > 0 else 0
        
        return {
            'total_loss': avg_total_loss,
            'ot_loss': avg_ot_loss,
            'kenergy': avg_kenergy,
            'reg_ratio': reg_ratio
        }
    else:
        return {'total_loss': 0, 'ot_loss': 0, 'kenergy': 0, 'reg_ratio': 0}

# ================================================================================================
# MAIN TRAINING LOOP
# ================================================================================================

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    """Main training function with integrated evaluation."""
    # Update CONFIG with Hydra configuration
    update_config_from_hydra(cfg)
    
    # Record startup time
    startup_start_time = time.time()
    
    # Generate run name based on wandb availability
    if CONFIG['use_wandb'] and not CONFIG['debug_mode']:
        run_name = None  # Will be set after wandb.init
    else:
        # Generate timestamp-based name
        run_name = f"train_v15_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Update log path with run name
    if run_name:
        log_dir = os.path.dirname(CONFIG['log_path'])
        log_filename = f"{run_name}.log"
        CONFIG['log_path'] = os.path.join(log_dir, log_filename)
    
    # Setup
    logger = setup_logging(CONFIG)
    if CONFIG['debug_mode']:
        logger.info("🐛 DEBUG MODE: Starting training with integrated evaluation (v15) - will exit after 1 epoch")
    else:
        logger.info("Starting training with integrated evaluation (v15)")
    
    # Log gradient clipping configuration
    if CONFIG['gradient_clipping']:
        logger.info(f"🎯 Gradient clipping configured: norm={CONFIG['clip_grad_norm']:.2f}, start_epoch={CONFIG['clip_start_epoch']}")
    else:
        logger.info("🎯 Gradient clipping disabled")
    
    # Load data
    data = load_data(CONFIG, logger)
    
    # Pre-calculate static radius for evaluation (only if evaluation is enabled)
    static_radius = None
    if CONFIG['enable_eval']:
        static_radius = precalculate_latent_radius(data, CONFIG, logger)
    
    # Setup model
    dyn_velocity = DynVelocity(CONFIG).to(CONFIG['device'])
    optimizer = torch.optim.AdamW(dyn_velocity.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    
    # Setup loss function
    loss_fn = setup_loss_function(CONFIG, logger)
    
    # Setup wandb (disable in debug mode)
    wandb_run = None
    if CONFIG['use_wandb'] and not CONFIG['debug_mode']:
        # Add evaluation config to wandb
        wandb_config = CONFIG.copy()
        wandb_config['static_radius'] = static_radius
        wandb_run = wandb.init(project=CONFIG['wandb_project'], config=wandb_config)
        run_name = wandb_run.name  # Get wandb-generated run name
        logger.info(f"Wandb logging initialized with run name: {run_name}")
        
        # Update log file with wandb run name
        log_dir = os.path.dirname(CONFIG['log_path'])
        new_log_path = os.path.join(log_dir, f"{run_name}.log")
        # Close current log handlers and reinitialize with new path
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        CONFIG['log_path'] = new_log_path
        logger = setup_logging(CONFIG)
        logger.info(f"Log file updated to use wandb run name: {run_name}")
        
    elif CONFIG['debug_mode']:
        logger.info("🐛 Debug mode: Wandb logging disabled")
    
    # Calculate and report startup time
    startup_end_time = time.time()
    startup_duration = startup_end_time - startup_start_time
    logger.info(f"⏱️ Startup completed in {startup_duration:.2f} seconds")
    
    # Training loop
    meter = DictAverageMeter()
    epoch = 0
    
    while epoch < CONFIG['n_epochs']:
        meter.reset()
        
        # Training step
        loss_metrics = train_step(dyn_velocity, optimizer, data, CONFIG, loss_fn, logger, epoch)
        
        # Debug mode: exit after first epoch
        if CONFIG['debug_mode'] and epoch >= 0:
            logger.info("🐛 Debug mode: Exiting after first epoch")
            break
        
        # Learning rate scheduling
        if CONFIG['lr_schedule'] is not None and epoch in CONFIG['lr_schedule']:
            new_lr = CONFIG['lr_schedule'][epoch]
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            logger.info(f"Learning rate changed to {new_lr:.2e} at epoch {epoch}")
        
        # Gradient clipping status notification
        if CONFIG['gradient_clipping'] and epoch == CONFIG['clip_start_epoch']:
            logger.info(f"🎯 Gradient clipping enabled at epoch {epoch} with norm threshold {CONFIG['clip_grad_norm']:.2f}")
        
        # Collect training metrics
        loss_dict = {
            "total_loss": loss_metrics['total_loss'],
            "ot_loss": loss_metrics['ot_loss'], 
            "kenergy": loss_metrics['kenergy'],
            "reg_ratio": loss_metrics['reg_ratio'],
            "lr": get_lr(optimizer),
            "gn": get_gn(dyn_velocity)
        }
        meter.update(loss_dict, n=1)
        
        epoch_loss_dict = meter.average()
        
        # Evaluation every eval_every epochs (if enabled) or always in debug mode
        if CONFIG['enable_eval'] and (epoch % CONFIG['eval_every'] == 0 or CONFIG['debug_mode']):
            eval_results = evaluate_model(dyn_velocity, data, CONFIG, static_radius, logger)
            
            # Add evaluation metrics to logging
            eval_dict = {}
            
            # Collect metrics for averaging
            expr_vel_norms = []
            pos_vel_norms = []
            coverages = []
            gw_distances = []
            label_consistencies = []
            
            for (start_idx, end_idx), metrics in eval_results.items():
                # Velocity norms
                if 'expr_velocity_norm_mean' in metrics:
                    eval_dict[f'eval_expr_vel_norm_{start_idx}_{end_idx}'] = metrics['expr_velocity_norm_mean']
                    expr_vel_norms.append(metrics['expr_velocity_norm_mean'])
                if 'pos_velocity_norm_mean' in metrics:
                    eval_dict[f'eval_pos_vel_norm_{start_idx}_{end_idx}'] = metrics['pos_velocity_norm_mean']
                    pos_vel_norms.append(metrics['pos_velocity_norm_mean'])
                
                # Other metrics
                if 'coverage' in metrics:
                    eval_dict[f'eval_coverage_{start_idx}_{end_idx}'] = metrics['coverage']
                    coverages.append(metrics['coverage'])
                if 'gw_distance' in metrics:
                    eval_dict[f'eval_gw_distance_{start_idx}_{end_idx}'] = metrics['gw_distance']
                    gw_distances.append(metrics['gw_distance'])
                if 'label_consistency' in metrics:
                    eval_dict[f'eval_label_consistency_{start_idx}_{end_idx}'] = metrics['label_consistency']
                    label_consistencies.append(metrics['label_consistency'])
            
            # Add averaged metrics (always compute for logging consistency)
            if expr_vel_norms:
                eval_dict['eval_expr_vel_norm_avg'] = np.mean(expr_vel_norms)
            if pos_vel_norms:
                eval_dict['eval_pos_vel_norm_avg'] = np.mean(pos_vel_norms)
            if coverages:
                eval_dict['eval_coverage_avg'] = np.mean(coverages)
            if gw_distances:
                eval_dict['eval_gw_distance_avg'] = np.mean(gw_distances)
            if label_consistencies:
                eval_dict['eval_label_consistency_avg'] = np.mean(label_consistencies)
            
            # Combine training and evaluation metrics
            combined_dict = {**epoch_loss_dict, **eval_dict}
            
            # Log to wandb
            if CONFIG['use_wandb'] and wandb_run:
                wandb.log(data=combined_dict, step=epoch)
            
            # Print evaluation results
            logger.info(f"Epoch {epoch} Evaluation:")
            for (start_idx, end_idx), metrics in eval_results.items():
                eval_str = f"  t{start_idx}→t{end_idx}: "
                if 'expr_velocity_norm_mean' in metrics:
                    eval_str += f"ExprVel={metrics['expr_velocity_norm_mean']:.2e} "
                if 'pos_velocity_norm_mean' in metrics:
                    eval_str += f"PosVel={metrics['pos_velocity_norm_mean']:.2e} "
                if 'coverage' in metrics:
                    eval_str += f"Coverage={metrics['coverage']*100:.0f}% "
                if 'gw_distance' in metrics:
                    eval_str += f"GW={metrics['gw_distance']:.2e} "
                if 'label_consistency' in metrics:
                    eval_str += f"LabelConsist={metrics['label_consistency']*100:.0f}%"
                logger.info(eval_str)
            
            # Print averaged results if multiple timepoint pairs exist
            if len(eval_results) > 1:
                avg_str = "  AVERAGE: "
                if expr_vel_norms:
                    avg_str += f"ExprVel={np.mean(expr_vel_norms):.2e} "
                if pos_vel_norms:
                    avg_str += f"PosVel={np.mean(pos_vel_norms):.2e} "
                if coverages:
                    avg_str += f"Coverage={np.mean(coverages)*100:.0f}% "
                if gw_distances:
                    avg_str += f"GW={np.mean(gw_distances):.2e} "
                if label_consistencies:
                    avg_str += f"LabelConsist={np.mean(label_consistencies)*100:.0f}%"
                logger.info(avg_str)
        
        else:
            # Log only training metrics
            if CONFIG['use_wandb'] and wandb_run:
                wandb.log(data=epoch_loss_dict, step=epoch)
        
        # Print training progress
        if epoch % CONFIG['print_every'] == 0:
            logger.info(f">>> Epoch{epoch}\t"
                       f"Loss={epoch_loss_dict['total_loss']:.3e}\t"
                       f"OT={epoch_loss_dict['ot_loss']:.3e}\t"
                       f"E={epoch_loss_dict['kenergy']:.3f}\t"
                       f"RegRatio={epoch_loss_dict['reg_ratio']:.3e}\t"
                       f"lr={epoch_loss_dict['lr']:.2e}\t"
                       f"gn={epoch_loss_dict['gn']:.2e}")
        
        # Save checkpoint
        if epoch % CONFIG['save_every'] == 0:
            model_path = os.path.join(CONFIG['save_path'], f'dyn_velocity-{run_name}-E{epoch}-v15.pt')
            optimizer_path = os.path.join(CONFIG['save_path'], f'optimizer-{run_name}-E{epoch}-v15.pt')
            torch.save(dyn_velocity.state_dict(), model_path)
            torch.save(optimizer.state_dict(), optimizer_path)
            logger.info(f"Checkpoint saved: {model_path}")
        
        # Increase epoch
        epoch += 1
    
    # Final evaluation (if enabled)
    if CONFIG['enable_eval']:
        logger.info("=== Final Evaluation ===")
        final_eval_results = evaluate_model(dyn_velocity, data, CONFIG, static_radius, logger)
        
        for (start_idx, end_idx), metrics in final_eval_results.items():
            final_str = f"Final t{start_idx}→t{end_idx}: "
            if 'expr_velocity_norm_mean' in metrics:
                final_str += f"ExprVel={metrics['expr_velocity_norm_mean']:.2e} "
            if 'pos_velocity_norm_mean' in metrics:
                final_str += f"PosVel={metrics['pos_velocity_norm_mean']:.2e} "
            if 'coverage' in metrics:
                final_str += f"Coverage={metrics['coverage']*100:.0f}% "
            if 'gw_distance' in metrics:
                final_str += f"GW={metrics['gw_distance']:.2e} "
            if 'label_consistency' in metrics:
                final_str += f"LabelConsist={metrics['label_consistency']*100:.0f}%"
            logger.info(final_str)
    
    # Finish
    if CONFIG['use_wandb'] and wandb_run:
        wandb_run.finish()
    
    logger.info("Training with evaluation completed successfully")

if __name__ == "__main__":
    main()
