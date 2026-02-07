"""
Visualization Module for CT-PET Image Analysis
"""

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Dict


def normalize(arr: np.ndarray, percentile_clip: float = 99.5) -> np.ndarray:
    """Normalize array to 0-1 range with percentile clipping."""
    arr = arr.astype(np.float32)
    low = np.percentile(arr, 100 - percentile_clip)
    high = np.percentile(arr, percentile_clip)
    arr = np.clip(arr, low, high)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)


def plot_slice(ct_slice: np.ndarray, pet_slice: Optional[np.ndarray] = None,
               mask_slice: Optional[np.ndarray] = None, title: str = "",
               alpha: float = 0.4, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plot a single slice with optional overlay."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.imshow(normalize(ct_slice), cmap='gray', aspect='equal')
    
    if pet_slice is not None:
        pet_norm = normalize(pet_slice)
        pet_masked = np.ma.masked_where(pet_norm < 0.1, pet_norm)
        ax.imshow(pet_masked, cmap='hot', alpha=alpha, aspect='equal')
    
    if mask_slice is not None and np.any(mask_slice > 0):
        ax.contour(mask_slice, levels=[0.5], colors=['lime'], linewidths=2)
    
    ax.set_title(title)
    ax.axis('off')
    return ax


def plot_fusion_comparison(ct_slice: np.ndarray, pet_original: np.ndarray,
                           pet_registered: np.ndarray) -> plt.Figure:
    """Plot fusion comparison before and after registration."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    ct_norm = normalize(ct_slice)
    
    axes[0].imshow(ct_norm, cmap='gray')
    axes[0].set_title('CT')
    axes[0].axis('off')
    
    axes[1].imshow(ct_norm, cmap='gray')
    axes[1].imshow(normalize(pet_original), cmap='hot', alpha=0.4)
    axes[1].set_title('Before Registration')
    axes[1].axis('off')
    
    axes[2].imshow(ct_norm, cmap='gray')
    axes[2].imshow(normalize(pet_registered), cmap='hot', alpha=0.4)
    axes[2].set_title('After Registration')
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig


def create_3d_montage(volume: np.ndarray, n_slices: int = 16,
                      pet_volume: Optional[np.ndarray] = None) -> plt.Figure:
    """Create a montage of slices through a 3D volume."""
    indices = np.linspace(0, volume.shape[0] - 1, n_slices, dtype=int)
    n_cols = int(np.ceil(np.sqrt(n_slices)))
    n_rows = int(np.ceil(n_slices / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 16))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        ax = axes[i]
        ax.imshow(normalize(volume[idx]), cmap='gray')
        if pet_volume is not None:
            pet_norm = normalize(pet_volume[idx])
            pet_masked = np.ma.masked_where(pet_norm < 0.1, pet_norm)
            ax.imshow(pet_masked, cmap='hot', alpha=0.4)
        ax.set_title(f'Slice {idx}', fontsize=8)
        ax.axis('off')
    
    for i in range(len(indices), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


def plot_orthogonal_views(ct_volume: np.ndarray, pet_volume: Optional[np.ndarray] = None,
                          center: Optional[Tuple[int, int, int]] = None) -> plt.Figure:
    """Plot orthogonal views (axial, coronal, sagittal)."""
    if center is None:
        center = tuple(s // 2 for s in ct_volume.shape)
    
    z, y, x = center
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    slices = [(ct_volume[z, :, :], 'Axial'),
              (ct_volume[:, y, :], 'Coronal'),
              (ct_volume[:, :, x], 'Sagittal')]
    
    pet_slices = None
    if pet_volume is not None:
        pet_slices = [pet_volume[z, :, :], pet_volume[:, y, :], pet_volume[:, :, x]]
    
    for i, ((ct_sl, title), ax) in enumerate(zip(slices, axes)):
        pet_sl = pet_slices[i] if pet_slices else None
        plot_slice(ct_sl, pet_sl, title=title, ax=ax)
    
    plt.tight_layout()
    return fig
