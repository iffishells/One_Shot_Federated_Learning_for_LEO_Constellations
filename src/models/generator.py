"""
Generator network for data-free knowledge distillation.

This module implements an improved Generator with residual-style blocks
that generates 224x224 images from random noise.
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Improved Generator network with residual-style blocks and better initialization.
    Generates 224x224 images from random noise for data-free knowledge distillation.
    
    Architecture:
        - Linear projection from latent space to initial feature map
        - 4 upsampling blocks: 14→28→56→112→224
        - Each block: ConvTranspose2d + BatchNorm + LeakyReLU + Conv2d + BatchNorm + LeakyReLU
    
    Args:
        latent_dim: Dimension of the input noise vector (default: 100)
        img_size: Output image size (default: 224)
        channels: Number of output channels (default: 3 for RGB)
        ngf: Base number of generator filters (default: 64)
    """

    def __init__(self, latent_dim: int = 100, img_size: int = 224, 
                 channels: int = 3, ngf: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.init_size = img_size // 16  # 14 for 224x224
        self.ngf = ngf

        # Project noise to initial feature map with more capacity
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, ngf * 8 * self.init_size * self.init_size),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Upsample blocks: 14→28→56→112→224
        self.up1 = self._make_up_block(ngf * 8, ngf * 4)  # 14→28
        self.up2 = self._make_up_block(ngf * 4, ngf * 2)  # 28→56
        self.up3 = self._make_up_block(ngf * 2, ngf)      # 56→112
        self.up4 = nn.Sequential(  # 112→224
            nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf // 2, channels, 3, 1, 1, bias=True),
            nn.Tanh()  # Output in [-1, 1]
        )

        # Initialize weights for better training
        self.apply(self._init_weights)

    def _make_up_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        """Create upsampling block with extra conv for more capacity."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def _init_weights(self, m: nn.Module):
        """Initialize weights using He initialization."""
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Generate images from noise vectors.
        
        Args:
            z: Noise tensor of shape (batch_size, latent_dim)
        
        Returns:
            Generated images of shape (batch_size, channels, img_size, img_size)
        """
        x = self.fc(z)
        x = x.view(x.size(0), self.ngf * 8, self.init_size, self.init_size)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return x

    def generate_batch(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Generate a batch of synthetic images.
        
        Args:
            batch_size: Number of images to generate
            device: Device to generate on
        
        Returns:
            Generated images tensor
        """
        z = torch.randn(batch_size, self.latent_dim, device=device)
        return self(z)

