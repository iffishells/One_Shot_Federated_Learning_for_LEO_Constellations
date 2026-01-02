"""
Classifier model factories for teacher and student networks.
"""

import torch.nn as nn
from torchvision import models


def make_resnet50(n_classes: int = 10) -> nn.Module:
    """
    Create a ResNet-50 model for classification.
    
    Args:
        n_classes: Number of output classes (default: 10)
    
    Returns:
        ResNet-50 model with modified final layer
    """
    model = models.resnet50(weights=None)  # No pretrain (offline-friendly)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


def make_resnet18(n_classes: int = 10) -> nn.Module:
    """
    Create a ResNet-18 model for classification (smaller student model).
    
    Args:
        n_classes: Number of output classes (default: 10)
    
    Returns:
        ResNet-18 model with modified final layer
    """
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


def make_model(model_name: str, n_classes: int = 10) -> nn.Module:
    """
    Factory function to create a model by name.
    
    Args:
        model_name: Name of the model ('ResNet50' or 'ResNet18')
        n_classes: Number of output classes
    
    Returns:
        The requested model
    
    Raises:
        ValueError: If model_name is not recognized
    """
    model_map = {
        "ResNet50": make_resnet50,
        "ResNet18": make_resnet18,
        "resnet50": make_resnet50,
        "resnet18": make_resnet18,
    }
    
    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_map.keys())}")
    
    return model_map[model_name](n_classes)

