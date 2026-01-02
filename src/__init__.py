"""
One-Shot Federated Learning for LEO Satellite Constellations
with Data-Free Knowledge Distillation

This package provides a modular implementation for:
- Non-IID data partitioning across satellite orbits
- Teacher model training on each orbit
- Data-free synthetic image generation
- Knowledge distillation to student model
"""

from .config import Config, KDConfig

__version__ = "1.0.0"

