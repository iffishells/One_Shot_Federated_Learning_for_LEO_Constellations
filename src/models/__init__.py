"""Model definitions for knowledge distillation."""

from .generator import Generator
from .classifiers import make_resnet50, make_resnet18

__all__ = ['Generator', 'make_resnet50', 'make_resnet18']

