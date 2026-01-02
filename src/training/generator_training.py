"""
Generator training for data-free knowledge distillation.

This module implements class-conditional DeepInversion-style optimization
to train a generator that creates synthetic images teachers agree on.
"""

import copy
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def train_generator(
    generator: nn.Module,
    teachers: List[nn.Module],
    orbit_labels: List[List[int]],
    device: torch.device,
    epochs: int = 500,
    batch_size: int = 32,
    lr: float = 2e-3,
    diversity_weight: float = 5.0,
    n_classes: int = 10
) -> nn.Module:
    """
    Train generator using class-conditional DeepInversion-style optimization.

    Key improvements over basic entropy minimization:
    1. Class-conditional generation - targets specific classes each epoch
    2. Total variation loss - encourages smooth, natural-looking images
    3. L2 regularization - prevents extreme pixel values
    4. Better loss weighting and best-state tracking

    Args:
        generator: Generator network
        teachers: List of trained teacher models
        orbit_labels: List of class labels each teacher knows
        device: Device to train on
        epochs: Number of training epochs
        batch_size: Batch size for generation
        lr: Learning rate
        diversity_weight: Weight for diversity loss
        n_classes: Number of classes

    Returns:
        Trained generator
    """
    generator = generator.to(device)
    generator.train()

    # Freeze teachers - they should not be updated
    for teacher in teachers:
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False

    optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    print("\n" + "=" * 60)
    print("PHASE 1: Training Generator (Class-Conditional DeepInversion)")
    print("=" * 60)
    print(f"Training for {epochs} epochs with batch_size={batch_size}")
    print("Target: confidence > 0.7, entropy < 0.5")

    avg_confidence = 0.0
    best_confidence = 0.0
    best_state = None

    epoch_pbar = tqdm(range(epochs), desc="Generator Training")
    for epoch in epoch_pbar:
        optimizer.zero_grad()

        # Generate images
        z = torch.randn(batch_size, generator.latent_dim, device=device)
        fake_images = generator(z)

        # =================================================================
        # CLASS-CONDITIONAL: Target a specific class this epoch for diversity
        # =================================================================
        target_class = epoch % n_classes

        # Collect predictions using class-aware ensemble
        vote_sum = torch.zeros(batch_size, n_classes, device=device)
        vote_count = torch.zeros(n_classes, device=device)

        for teacher, labels in zip(teachers, orbit_labels):
            logits = teacher(fake_images)
            probs = F.softmax(logits, dim=1)
            for c in labels:
                vote_sum[:, c] = vote_sum[:, c] + probs[:, c]
                vote_count[c] = vote_count[c] + 1

        vote_count = vote_count.clamp(min=1)
        avg_pred = vote_sum / vote_count.unsqueeze(0)
        avg_pred = avg_pred / (avg_pred.sum(dim=1, keepdim=True) + 1e-8)

        # =================================================================
        # LOSS 1: Cross-entropy to target class (class-conditional)
        # =================================================================
        target_labels = torch.full(
            (batch_size,), target_class, device=device, dtype=torch.long
        )
        loss_target = F.cross_entropy(torch.log(avg_pred + 1e-8), target_labels)

        # =================================================================
        # LOSS 2: Entropy minimization (confidence)
        # =================================================================
        entropy = -(avg_pred * torch.log(avg_pred + 1e-8)).sum(dim=1)
        loss_entropy = entropy.mean()

        # =================================================================
        # LOSS 3: One-hot encouragement (maximize confidence)
        # =================================================================
        max_probs = avg_pred.max(dim=1)[0]
        loss_onehot = -max_probs.mean()

        # =================================================================
        # LOSS 4: Total Variation (smoothness - helps generate cleaner images)
        # =================================================================
        tv_h = torch.abs(fake_images[:, :, 1:, :] - fake_images[:, :, :-1, :]).mean()
        tv_w = torch.abs(fake_images[:, :, :, 1:] - fake_images[:, :, :, :-1]).mean()
        loss_tv = tv_h + tv_w

        # =================================================================
        # LOSS 5: L2 regularization (prevent extreme values)
        # =================================================================
        loss_l2 = (fake_images ** 2).mean()

        # =================================================================
        # LOSS 6: Feature statistics matching
        # =================================================================
        img_mean = fake_images.mean()
        img_std = fake_images.std()
        loss_stats = (img_mean ** 2) + ((img_std - 1.0) ** 2)

        # =================================================================
        # TOTAL LOSS - Weighted combination
        # =================================================================
        total_loss = (
            1.0 * loss_target +      # Class targeting
            1.0 * loss_entropy +     # Entropy minimization
            3.0 * loss_onehot +      # Confidence maximization (increased weight)
            0.0005 * loss_tv +       # Smoothness
            0.005 * loss_l2 +        # L2 regularization
            0.05 * loss_stats        # Feature matching
        )

        # Backward and update
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Track best state
        avg_confidence = max_probs.mean().item()
        if avg_confidence > best_confidence:
            best_confidence = avg_confidence
            best_state = copy.deepcopy(generator.state_dict())

        loss_val = total_loss.item()
        entropy_val = loss_entropy.item()

        # Clear memory
        del fake_images, vote_sum, avg_pred, total_loss
        torch.cuda.empty_cache()

        # Logging
        epoch_pbar.set_postfix(
            loss=f"{loss_val:.3f}",
            entropy=f"{entropy_val:.2f}",
            conf=f"{avg_confidence:.2f}",
            best=f"{best_confidence:.2f}",
            cls=target_class
        )

        if (epoch + 1) % 50 == 0:
            tqdm.write(
                f"Generator Epoch {epoch + 1}: loss={loss_val:.4f}, "
                f"entropy={entropy_val:.3f}, conf={avg_confidence:.3f}, "
                f"best={best_confidence:.3f}, lr={scheduler.get_last_lr()[0]:.6f}"
            )

        # Early stopping if confidence is good
        if avg_confidence > 0.80 and entropy_val < 0.4:
            tqdm.write(
                f"✓ Early stopping: confidence={avg_confidence:.3f}, "
                f"entropy={entropy_val:.3f}"
            )
            break

    # Load best generator state
    if best_state is not None and best_confidence > avg_confidence:
        generator.load_state_dict(best_state)
        print(f"Loaded best generator state (conf={best_confidence:.3f})")

    generator.eval()

    # Restore teachers' requires_grad
    for teacher in teachers:
        for param in teacher.parameters():
            param.requires_grad = True

    print(f"\nGenerator training complete!")
    print(f"Final confidence: {max(avg_confidence, best_confidence):.3f}")

    if best_confidence < 0.5:
        print("⚠️ WARNING: Generator confidence is low.")
        print("   Data-free KD is challenging - synthetic data may be suboptimal.")
        print("   For better results, consider using real data (use_synthetic_data=False)")

    return generator

