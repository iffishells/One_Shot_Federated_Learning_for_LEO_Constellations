"""
Synthetic data generation utilities for data-free knowledge distillation.
"""

import torch
from typing import Iterator, Tuple


class SyntheticDataLoader:
    """
    A DataLoader-like object that generates synthetic batches on-the-fly
    using a trained generator.
    
    This enables truly data-free knowledge distillation where the ground station
    has no access to real data - only the uploaded teacher models.
    
    Args:
        generator: Trained Generator network
        num_batches: Number of batches to generate per epoch
        batch_size: Batch size for generation
        device: Device to generate on
    
    Usage:
        loader = SyntheticDataLoader(generator, num_batches=100, batch_size=64, device='cuda')
        for images, _ in loader:
            # images are synthetic, labels are dummy (not used in KD)
            ...
    """

    def __init__(self, generator, num_batches: int, batch_size: int, device: torch.device):
        self.generator = generator
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.device = device

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Iterate over synthetic batches.
        
        Yields:
            Tuple of (synthetic_images, dummy_labels)
        """
        self.generator.eval()
        for _ in range(self.num_batches):
            with torch.no_grad():
                fake_images = self.generator.generate_batch(self.batch_size, self.device)
            # Yield images with dummy labels (not used in KD)
            yield fake_images, torch.zeros(self.batch_size, dtype=torch.long, device=self.device)

    def __len__(self) -> int:
        """Return number of batches."""
        return self.num_batches

