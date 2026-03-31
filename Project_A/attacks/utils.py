"""
Utility functions for attacks
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Dict, Optional
import random

def load_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    """
    Load model from checkpoint
    Assumes model architecture is defined elsewhere
    """
    # 这里需要根据你的模型架构导入
    # 临时解决方案：你需要传入模型类
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    return model

def select_target_classes(
    labels: torch.Tensor,
    strategy: str = 'random',
    num_classes: int = 10,
    exclude_original: bool = True
) -> torch.Tensor:
    """
    Select target classes for best/avg/worst case evaluation
    
    Args:
        labels: true labels
        strategy: 'best', 'avg', 'worst', or 'random'
        num_classes: number of classes
        exclude_original: whether to exclude original class
    
    Returns:
        target classes
    """
    batch_size = labels.size(0)
    targets = torch.zeros_like(labels)
    
    if strategy == 'random' or strategy == 'avg':
        # Randomly select targets (excluding original)
        for i in range(batch_size):
            possible = list(range(num_classes))
            if exclude_original:
                possible.remove(labels[i].item())
            targets[i] = random.choice(possible)
    
    elif strategy == 'best':
        # For best case, we'll let the attack try all targets
        # and report the easiest one
        # This function just returns placeholder, actual best case
        # needs to be handled by running all targets
        targets = torch.ones_like(labels) * -1
    
    elif strategy == 'worst':
        # Similar to best case
        targets = torch.ones_like(labels) * -1
    
    return targets

def compute_metrics(results: List[Dict]) -> Dict:
    """
    Compute summary metrics from attack results
    """
    successful = [r for r in results if r['success']]
    
    if not successful:
        return {
            'success_rate': 0.0,
            'mean_l2': float('inf'),
            'median_l2': float('inf'),
            'std_l2': float('inf'),
            'min_l2': float('inf'),
            'max_l2': float('inf'),
            'total_samples': len(results),
            'successful_samples': 0
        }
    
    l2_distances = [r['l2_distance'] for r in successful]
    
    return {
        'success_rate': len(successful) / len(results) * 100,
        'mean_l2': np.mean(l2_distances),
        'median_l2': np.median(l2_distances),
        'std_l2': np.std(l2_distances),
        'min_l2': np.min(l2_distances),
        'max_l2': np.max(l2_distances),
        'total_samples': len(results),
        'successful_samples': len(successful)
    }

def visualize_attack(
    image: torch.Tensor,
    adversarial: torch.Tensor,
    true_label: int,
    pred_label: int,
    l2_dist: float,
    save_path: Optional[str] = None
):
    """
    Visualize original image, adversarial image, and perturbation
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Original image
    img = image.cpu().numpy().squeeze()
    if img.ndim == 3:
        img = img.transpose(1, 2, 0)
    axes[0].imshow(img, cmap='gray' if img.ndim == 2 else None)
    axes[0].set_title(f'Original\nLabel: {true_label}')
    axes[0].axis('off')
    
    # Adversarial image
    adv = adversarial.cpu().numpy().squeeze()
    if adv.ndim == 3:
        adv = adv.transpose(1, 2, 0)
    axes[1].imshow(adv, cmap='gray' if adv.ndim == 2 else None)
    axes[1].set_title(f'Adversarial\nPred: {pred_label}\nL2: {l2_dist:.4f}')
    axes[1].axis('off')
    
    # Perturbation (amplified for visualization)
    pert = adversarial - image
    pert = pert.cpu().numpy().squeeze()
    if pert.ndim == 3:
        pert = pert.transpose(1, 2, 0)
    # Amplify perturbation for visibility
    pert_vis = (pert * 10).clip(0, 1)
    axes[2].imshow(pert_vis, cmap='gray' if pert_vis.ndim == 2 else None)
    axes[2].set_title(f'Perturbation\n(amplified 10x)')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()