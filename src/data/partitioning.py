"""
Non-IID data partitioning utilities for federated learning across satellite orbits.
"""

import random
from typing import List, Set
import torch
from torch.utils.data import Subset, Dataset


def idxs_for_labels(dataset: Dataset, labels: List[int]) -> List[int]:
    """
    Get indices of samples with the specified labels.
    
    Uses dataset.targets for fast label-based indexing (avoids loading images).
    
    Args:
        dataset: PyTorch dataset with .targets attribute
        labels: List of label indices to filter for
    
    Returns:
        List of sample indices
    """
    labels_set: Set[int] = set(labels)
    targets = dataset.targets
    if isinstance(targets, torch.Tensor):
        targets = targets.tolist()
    return [i for i, y in enumerate(targets) if y in labels_set]


def subset_by_labels(dataset: Dataset, labels: List[int], 
                     limit: int = None, fast_mode: bool = False) -> Subset:
    """
    Create a subset of the dataset containing only specified labels.
    
    Args:
        dataset: PyTorch dataset
        labels: List of labels to include
        limit: Maximum number of samples (optional)
        fast_mode: Whether to apply fast mode limit
    
    Returns:
        Subset of the dataset
    """
    idxs = idxs_for_labels(dataset, labels)
    if fast_mode and limit:
        idxs = idxs[:limit]
    return Subset(dataset, idxs)


def create_orbit_splits(
    train_dataset: Dataset,
    orbit_labels: List[List[int]],
    train_val_split: float = 0.85,
    fast_mode: bool = False,
    fast_train_size: int = 800,
    fast_val_size: int = 200
) -> tuple:
    """
    Split the training dataset into orbit-specific train/val subsets.
    
    This implements non-IID partitioning where each orbit (satellite constellation)
    only has access to a subset of classes, mimicking real-world LEO satellite scenarios.
    
    Args:
        train_dataset: Full training dataset
        orbit_labels: List of class labels for each orbit
        train_val_split: Fraction of data for training (rest for validation)
        fast_mode: Whether to use reduced data sizes for quick testing
        fast_train_size: Training size per orbit in fast mode
        fast_val_size: Validation size per orbit in fast mode
    
    Returns:
        Tuple of (train_orbit_subsets, val_orbit_subsets) - lists of Subsets
    """
    train_orbit_subsets = []
    val_orbit_subsets = []
    
    for lbls in orbit_labels:
        idxs = idxs_for_labels(train_dataset, lbls)
        random.shuffle(idxs)
        
        split_idx = int(train_val_split * len(idxs))
        tr_idx = idxs[:split_idx]
        va_idx = idxs[split_idx:]
        
        if fast_mode:
            tr_idx = tr_idx[:fast_train_size]
            va_idx = va_idx[:fast_val_size]
        
        train_orbit_subsets.append(Subset(train_dataset, tr_idx))
        val_orbit_subsets.append(Subset(train_dataset, va_idx))
    
    return train_orbit_subsets, val_orbit_subsets


def print_orbit_info(orbit_labels: List[List[int]], 
                     train_subsets: List[Subset], 
                     val_subsets: List[Subset],
                     class_names: List[str]):
    """
    Print information about orbit data distribution.
    
    Args:
        orbit_labels: List of class labels for each orbit
        train_subsets: Training subsets for each orbit
        val_subsets: Validation subsets for each orbit
        class_names: Names of each class
    """
    print("\nOrbit class assignments:")
    for i, labels in enumerate(orbit_labels, 1):
        names = [class_names[l] for l in labels]
        print(f"  Orbit {i}: {names}")
    
    print("\nOrbit data sizes:")
    for i, (tr, va) in enumerate(zip(train_subsets, val_subsets), 1):
        print(f"  Orbit {i}: labels {orbit_labels[i-1]}, "
              f"train {len(tr)}, val {len(va)}")

