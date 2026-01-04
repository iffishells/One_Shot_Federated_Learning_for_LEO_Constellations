"""
Generator training for data-free knowledge distillation.

This module implements class-conditional DeepInversion-style optimization
to train a generator that creates synthetic images teachers agree on.

Uses:
- Logits (pre-softmax outputs) from teacher models
- BatchNorm statistics matching
- Ensemble predictions via averaging logits
- KL divergence regularization
"""

import copy
from typing import List, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def extract_bn_statistics(model: nn.Module) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Extract BatchNorm layer statistics (running_mean and running_var) from a model.
    
    These statistics help the generator mimic the distribution of real data
    and stabilize training by matching feature distributions.
    
    Args:
        model: PyTorch model (may be wrapped in DataParallel)
    
    Returns:
        Dictionary mapping layer names to (mean, variance) tuples
    """
    # Unwrap DataParallel if needed
    actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    
    bn_stats = {}
    for name, module in actual_model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            # Get running statistics (channel-wise means and variances)
            running_mean = module.running_mean.clone().detach()
            running_var = module.running_var.clone().detach()
            bn_stats[name] = (running_mean, running_var)
    
    return bn_stats


def compute_bn_loss(fake_images: torch.Tensor, 
                    teachers: List[nn.Module],
                    device: torch.device) -> torch.Tensor:
    """
    Compute BatchNorm statistics matching loss.
    
    Compares the BatchNorm statistics of synthetic images passed through
    teacher models with the stored running statistics. This helps the generator
    mimic the distribution of real satellite data.
    
    Note: To avoid inplace operation issues, we temporarily set BatchNorm
    to training mode to compute batch statistics, then restore eval mode.
    
    Args:
        fake_images: Generated synthetic images (requires_grad=True)
        teachers: List of teacher models
        device: Device for computation
    
    Returns:
        BatchNorm matching loss
    """
    total_loss = 0.0
    count = 0
    
    for teacher in teachers:
        # Get BatchNorm statistics from teacher (running means and variances)
        bn_stats = extract_bn_statistics(teacher)
        
        if not bn_stats:
            continue
        
        # Hook to capture BatchNorm layer inputs (before normalization)
        bn_inputs = {}
        hooks = []
        bn_modules = {}  # Store BN modules to temporarily switch to train mode
        
        def make_hook(name):
            def hook(module, input, output):
                if isinstance(module, nn.BatchNorm2d):
                    # Capture input before BatchNorm processes it
                    if len(input) > 0 and input[0].requires_grad:
                        x = input[0]  # Don't detach - keep gradient flow
                        if x.dim() == 4:  # [B, C, H, W]
                            # Compute batch statistics from input (before BN normalization)
                            # Channel-wise mean and variance
                            batch_mean = x.mean(dim=[0, 2, 3])  # [C]
                            # Use unbiased=False to match PyTorch's running_var computation
                            batch_var = x.var(dim=[0, 2, 3], unbiased=False)  # [C]
                            bn_inputs[name] = (batch_mean, batch_var)
            return hook
        
        # Register hooks and store BN modules
        actual_teacher = teacher.module if isinstance(teacher, torch.nn.DataParallel) else teacher
        for name, module in actual_teacher.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                hooks.append(module.register_forward_hook(make_hook(name)))
                bn_modules[name] = module
        
        try:
            # Temporarily set BatchNorm to training mode to compute batch statistics
            # This avoids inplace operation issues with running statistics
            bn_training_states = {}
            for name, module in bn_modules.items():
                bn_training_states[name] = module.training
                module.train()  # Set to training mode to compute batch stats
            
            # Forward pass with gradients enabled
            _ = teacher(fake_images)
            
            # Restore BatchNorm states
            for name, module in bn_modules.items():
                module.train(bn_training_states[name])
            
            # Compare batch statistics with running statistics
            for name, (batch_mean, batch_var) in bn_inputs.items():
                if name in bn_stats:
                    running_mean, running_var = bn_stats[name]
                    # Match means and variances (MSE loss)
                    # Ensure tensors are on same device
                    running_mean = running_mean.to(device)
                    running_var = running_var.to(device)
                    mean_loss = F.mse_loss(batch_mean, running_mean)
                    var_loss = F.mse_loss(batch_var, running_var)
                    total_loss += mean_loss + var_loss
                    count += 1
        except (RuntimeError, ValueError) as e:
            # If there's an inplace operation issue, skip this teacher
            # Restore states even if there was an error
            for name, module in bn_modules.items():
                if name in bn_training_states:
                    module.train(bn_training_states[name])
            pass
        finally:
            # Always remove hooks
            for hook in hooks:
                hook.remove()
    
    if count == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    return total_loss / count


def train_generator(
    generator: nn.Module,
    teachers: List[nn.Module],
    orbit_labels: List[List[int]],
    device: torch.device,
    epochs: int = 500,
    batch_size: int = 32,
    lr: float = 2e-3,
    diversity_weight: float = 5.0,
    n_classes: int = None,
    use_multi_gpu: bool = False
) -> nn.Module:
    """
    Train generator using class-conditional DeepInversion-style optimization.

    Implementation follows the paper's methodology:
    1. Uses logits (pre-softmax outputs) from teachers for rich information
    2. Ensemble predictions: D(x̂) = (1/|M|) Σ f_l(x̂, w_Kl) - averages logits
    3. BatchNorm statistics matching - uses channel-wise means/variances from teachers
    4. Class-conditional generation - targets specific classes each epoch
    5. Total variation loss - encourages smooth, natural-looking images
    6. L2 regularization - prevents extreme pixel values

    Args:
        generator: Generator network
        teachers: List of trained teacher models
        orbit_labels: List of class labels each teacher knows
        device: Device to train on
        epochs: Number of training epochs
        batch_size: Batch size for generation
        lr: Learning rate
        diversity_weight: Weight for diversity loss (currently unused, kept for compatibility)
        n_classes: Number of classes (inferred from orbit_labels if None)

    Returns:
        Trained generator
    """
    # Infer n_classes from orbit_labels if not provided
    if n_classes is None:
        all_labels = set()
        for labels in orbit_labels:
            all_labels.update(labels)
        n_classes = max(all_labels) + 1 if all_labels else 10
        print(f"Inferred n_classes={n_classes} from orbit_labels")
    
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
    print(f"Number of classes: {n_classes}")
    print("Target: confidence > 0.7, entropy < 0.5")

    avg_confidence = 0.0
    best_confidence = 0.0
    best_state = None

    # Get the actual generator module (unwrap DataParallel if needed)
    actual_generator = generator.module if isinstance(generator, torch.nn.DataParallel) else generator
    
    epoch_pbar = tqdm(range(epochs), desc="Generator Training")
    for epoch in epoch_pbar:
        optimizer.zero_grad()

        # Generate images
        z = torch.randn(batch_size, actual_generator.latent_dim, device=device)
        fake_images = generator(z)

        # =================================================================
        # CLASS-CONDITIONAL: Target a specific class this epoch for diversity
        # =================================================================
        target_class = epoch % n_classes

        # =================================================================
        # ENSEMBLE PREDICTIONS: Average logits across teachers (not probabilities)
        # D(x̂) = (1/|M|) Σ f_l(x̂, w_Kl) - using logits for rich information
        # =================================================================
        logit_sum = torch.zeros(batch_size, n_classes, device=device)
        vote_count = torch.zeros(n_classes, device=device)

        for teacher, labels in zip(teachers, orbit_labels):
            logits = teacher(fake_images)  # Get logits (pre-softmax outputs)
            for c in labels:
                logit_sum[:, c] = logit_sum[:, c] + logits[:, c]
                vote_count[c] = vote_count[c] + 1

        # Average logits per class (ensemble of logits)
        vote_count = vote_count.clamp(min=1)
        avg_logits = logit_sum / vote_count.unsqueeze(0)
        
        # Convert to probabilities for loss computation
        avg_pred = F.softmax(avg_logits, dim=1)

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
        # LOSS 6: BatchNorm Statistics Matching
        # Match channel-wise means and variances from teacher BatchNorm layers
        # Note: This can be computationally expensive and may have gradient issues
        # with some model architectures, so we wrap it in try-except
        # =================================================================
        try:
            loss_bn = compute_bn_loss(fake_images, teachers, device)
            if not loss_bn.requires_grad:
                loss_bn = loss_bn.requires_grad_(True)
        except (RuntimeError, ValueError) as e:
            # Fallback if BatchNorm computation fails (e.g., inplace operation issues)
            # This can happen with certain model architectures or when using DataParallel
            loss_bn = torch.tensor(0.0, device=device, requires_grad=True)

        # =================================================================
        # LOSS 7: Image-level feature statistics matching (fallback)
        # =================================================================
        img_mean = fake_images.mean()
        img_std = fake_images.std()
        loss_stats = (img_mean ** 2) + ((img_std - 1.0) ** 2)

        # =================================================================
        # TOTAL LOSS - Weighted combination
        # =================================================================
        # Better weights for visual quality
        total_loss = (
                2.0 * loss_target +  # Stronger class targeting
                0.5 * loss_entropy +  # Less entropy focus
                1.0 * loss_onehot +  # Reduced one-hot
                0.01 * loss_tv +  # More smoothness
                0.01 * loss_l2 +  # More regularization
                0.1 * loss_stats +  # Better statistics
                1.0 * bn_loss  # ADD: BatchNorm matching (critical!)
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
        del fake_images, logit_sum, avg_logits, avg_pred, total_loss
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


def get_bn_statistics_loss(teachers, fake_images):
    """Match BatchNorm running statistics from teachers (DeepInversion)."""
    bn_loss = 0.0
    n_bn = 0

    for teacher in teachers:
        # Forward pass to capture BN statistics
        _ = teacher(fake_images)

        for module in teacher.modules():
            if isinstance(module, nn.BatchNorm2d):
                # Running mean/var are the target statistics
                # Current batch mean/var should match
                if hasattr(module, 'running_mean') and module.running_mean is not None:
                    # Get current batch statistics from the hook
                    # BatchNorm stores them during forward
                    mean_loss = (module.running_mean - fake_images.mean([0, 2, 3][:module.running_mean.dim()])).pow(
                        2).mean()
                    var_loss = (module.running_var - fake_images.var([0, 2, 3][:module.running_var.dim()])).pow(
                        2).mean()
                    bn_loss += mean_loss + var_loss
                    n_bn += 1

    return bn_loss / max(n_bn, 1)

