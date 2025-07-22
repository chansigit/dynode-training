"""
Utility functions for DynVelocity training.
"""

import os
import logging
import torch


def setup_logging(config):
    """Setup logging configuration with configurable log path."""
    # Ensure log directory exists
    log_path = config['log_path']
    log_dir = os.path.dirname(log_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"Created log directory: {log_dir}")
    
    # Setup logging to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured: {log_path}")
    return logger


def get_gn(model, norm_type=2):
    """Calculate gradient norm."""
    grads = [p.grad.detach().norm(norm_type) for p in model.parameters() if p.grad is not None]
    if len(grads) == 0:
        return 0.0
    total_norm = torch.norm(torch.stack(grads), norm_type)
    return total_norm.item()


def get_lr(optimizer):
    """Get current learning rate."""
    return [group['lr'] for group in optimizer.param_groups][0]


class DictAverageMeter:
    """Track averages of multiple metrics."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.sums = dict()
        self.counts = dict()

    def update(self, metric_dict, n=1):
        for k, v in metric_dict.items():
            if k not in self.sums:
                self.sums[k] = 0.0
                self.counts[k] = 0
            self.sums[k] += v * n
            self.counts[k] += n

    def average(self):
        return {k: self.sums[k] / self.counts[k] for k in self.sums}