"""
Cell dynamics model for neural ODE-based cell trajectory modeling.
Contains the DynVelocity model implementation.
"""

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.utils.parametrizations import spectral_norm

from dynamica.sat import SpatialAttentionLayer
from dynamica.equi import E3NNVelocityPredictor


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
        self.augmented_dim = config['augmented_dim']
        self.sigma = config['sigma']
        self.message_passing = config['message_passing']
        self.static_pos = config['static_pos']
        self.energy_regularization = config['energy_regularization']
        self.expr_autonomous = config['expr_autonomous']
        self.pos_autonomous = config['pos_autonomous']

        # Base MLP for expression features (now includes auxiliary features)
        base_input_dim = self.input_dim + self.position_dim + self.augmented_dim
        self.mlp_base = nn.Sequential(
            spectral_norm(nn.Linear(base_input_dim, self.hidden_dim)), 
            nn.LayerNorm(self.hidden_dim), nn.SiLU(), 
            spectral_norm(nn.Linear(self.hidden_dim, self.hidden_dim)), 
            nn.LayerNorm(self.hidden_dim), nn.SiLU(), 
            spectral_norm(nn.Linear(self.hidden_dim, self.hidden_dim)), 
            nn.LayerNorm(self.hidden_dim), nn.SiLU(), 
            spectral_norm(nn.Linear(self.hidden_dim, self.hidden_dim))
        )
        
        # Spatial attention layers (optional)
        if self.message_passing:
            # SpatialAttentionLayer input_dim parameter is for features only (not including positions)
            spatial_feature_dim = self.input_dim + self.augmented_dim
            self.spatial_base1 = nn.Sequential(
                SpatialAttentionLayer(
                    input_dim=spatial_feature_dim, p_dim=self.position_dim,
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

        # Augmented velocity predictor (for augmented dimensions)
        if self.augmented_dim > 0:
            aug_input_dim = self.hidden_dim * 2 + (0 if self.expr_autonomous else 1)
            self.aug_head = nn.Sequential(
                spectral_norm(nn.Linear(aug_input_dim, self.hidden_dim)),
                nn.LayerNorm(self.hidden_dim), nn.SiLU(),
                spectral_norm(nn.Linear(self.hidden_dim, self.augmented_dim))
            )

        self.apply(self._init_weights)

    def forward(self, t, state_tensor, args=None):
        """Forward pass through the model."""
        # Parse augmented state tensor: [expr, pos, aug, energy]
        expr_dim = self.input_dim
        pos_dim = self.position_dim
        aug_dim = self.augmented_dim
        
        # Remove energy dimension if present
        if self.energy_regularization:
            features = state_tensor[:, :-1]  # Remove last energy dimension
        else:
            features = state_tensor
        
        # Split features: [expr, pos, aug]
        expr_features = features[:, :expr_dim]
        pos_features = features[:, expr_dim:expr_dim + pos_dim]
        
        # Handle augmented features
        if aug_dim > 0:
            aug_features = features[:, expr_dim + pos_dim:expr_dim + pos_dim + aug_dim]
        else:
            aug_features = torch.empty(expr_features.size(0), 0, device=expr_features.device)
        
        # Combine all features for input (expr + pos + aug)
        X = torch.cat([expr_features, pos_features, aug_features], dim=1)

        # Compute base representation
        H0 = self.mlp_base(X)
        
        # Compute spatial representation (if enabled)
        if self.message_passing:
            # SpatialAttentionLayer expects [features, positions] concatenated
            spatial_features = torch.cat([expr_features, aug_features], dim=1)
            spatial_input = torch.cat([spatial_features, pos_features], dim=1)
            H1 = self.spatial_base1(spatial_input)
            # spatial_base2 also needs positions concatenated
            spatial_input2 = torch.cat([H1, pos_features], dim=1)
            H1 = self.spatial_base2(spatial_input2)
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

        # Predict augmented velocities
        if aug_dim > 0:
            if self.expr_autonomous:
                aug_velo = self.aug_head(H)
            else:
                t_expanded = t.expand(H.size(0), 1)
                aug_velo = self.aug_head(torch.cat([H, t_expanded], dim=1))
        else:
            aug_velo = torch.empty(expr_features.size(0), 0, device=expr_features.device)

        # Combine velocities: [expr_velo, pos_velo, aug_velo]
        velocity = torch.cat([feature_velo, pos_velo, aug_velo], dim=1)
        
        # Add energy if regularization is enabled
        if self.energy_regularization:
            kinetic_energy = 0.5 * (feature_velo ** 2).mean(dim=1) + 0.5 * (pos_velo ** 2).mean(dim=1)
            if aug_dim > 0:
                kinetic_energy += 0.5 * (aug_velo ** 2).mean(dim=1)
            velocity = torch.cat([velocity, kinetic_energy.unsqueeze(1)], dim=1)
            
        return velocity