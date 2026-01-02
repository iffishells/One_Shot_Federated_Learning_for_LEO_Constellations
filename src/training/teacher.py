"""
Teacher model training utilities.
"""

import copy
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


def train_teacher(
    model: nn.Module,
    train_subset: Subset,
    val_subset: Subset,
    device: torch.device,
    epochs: int = 300,
    lr: float = 1e-3,
    batch_size: int = 32,
    early_stopping: bool = True,
    patience: int = 10,
    min_delta: float = 0.001,
    num_workers: int = 2
) -> Tuple[nn.Module, int, float]:
    """
    Train a teacher model with optional early stopping.
    
    Args:
        model: Teacher model to train
        train_subset: Training data subset
        val_subset: Validation data subset
        device: Device to train on
        epochs: Maximum number of epochs
        lr: Learning rate
        batch_size: Batch size
        early_stopping: Whether to use early stopping
        patience: Early stopping patience (epochs without improvement)
        min_delta: Minimum improvement to reset patience counter
        num_workers: Number of data loader workers
    
    Returns:
        Tuple of (trained_model, epochs_trained, best_val_accuracy)
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, 
        num_workers=num_workers, pin_memory=True
    )

    # Early stopping setup
    best_val_acc = 0.0
    best_state = None
    patience_counter = 0
    epochs_trained = 0

    epoch_pbar = tqdm(range(epochs), desc="Training", leave=False)
    for ep in epoch_pbar:
        # Training phase
        model.train()
        total = 0
        correct = 0
        loss_sum = 0.0
        
        batch_pbar = tqdm(train_loader, desc=f"Epoch {ep + 1}/{epochs}", leave=False)
        for x, y in batch_pbar:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_sum += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
            batch_pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{correct / total:.3f}")
        
        train_acc = correct / total
        train_loss = loss_sum / total
        
        # Validation phase
        model.eval()
        val_total = 0
        val_correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                val_correct += (out.argmax(1) == y).sum().item()
                val_total += x.size(0)

        val_acc = val_correct / val_total
        epochs_trained = ep + 1
        
        epoch_pbar.set_postfix(
            train_acc=f"{train_acc:.3f}", 
            val_acc=f"{val_acc:.3f}", 
            best=f"{best_val_acc:.3f}"
        )
        tqdm.write(
            f"Teacher ep {ep + 1}: train loss {train_loss:.3f}, "
            f"train acc {train_acc:.3f}, val acc {val_acc:.3f}"
        )

        # Early stopping check
        if early_stopping:
            if val_acc > best_val_acc + min_delta:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    tqdm.write(
                        f"  ⚠ Early stopping at epoch {ep + 1}! "
                        f"Best val acc: {best_val_acc:.4f}"
                    )
                    if best_state is not None:
                        model.load_state_dict(best_state)
                    break
        else:
            # Just track best without early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())

    # Load best model at the end
    if best_state is not None and not early_stopping:
        model.load_state_dict(best_state)

    return model.eval(), epochs_trained, best_val_acc


def train_all_teachers(
    teacher_factory,
    train_subsets: list,
    val_subsets: list,
    orbit_labels: list,
    device: torch.device,
    epochs: int = 300,
    lr: float = 0.001,
    batch_size: int = 32,
    early_stopping: bool = True,
    patience: int = 10,
    min_delta: float = 0.001
) -> Tuple[list, list]:
    """
    Train all teacher models for each orbit.
    
    Args:
        teacher_factory: Function to create a new teacher model
        train_subsets: Training subsets for each orbit
        val_subsets: Validation subsets for each orbit
        orbit_labels: Class labels for each orbit
        device: Device to train on
        epochs: Maximum training epochs per teacher
        lr: Learning rate
        batch_size: Batch size
        early_stopping: Whether to use early stopping
        patience: Early stopping patience
        min_delta: Minimum improvement delta
    
    Returns:
        Tuple of (teachers_list, training_info_list)
    """
    teachers = []
    teacher_training_info = []
    
    print(f"\nTraining teachers (epochs={epochs}, lr={lr}, bs={batch_size})...")
    if early_stopping:
        print(f"Early stopping enabled: patience={patience}, min_delta={min_delta}")

    teacher_pbar = tqdm(
        enumerate(zip(train_subsets, val_subsets), 1),
        total=len(train_subsets), 
        desc="Teachers"
    )
    
    for i, (tr, va) in teacher_pbar:
        teacher_pbar.set_description(
            f"Teacher {i}/{len(train_subsets)} (Orbit labels: {orbit_labels[i - 1]})"
        )
        
        teacher = teacher_factory()
        teacher, epochs_trained, best_acc = train_teacher(
            teacher, tr, va,
            device=device,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            early_stopping=early_stopping,
            patience=patience,
            min_delta=min_delta
        )
        
        teachers.append(teacher)
        teacher_training_info.append({
            "epochs_trained": epochs_trained,
            "best_val_acc": best_acc,
            "early_stopped": epochs_trained < epochs
        })
        
        tqdm.write(
            f"Teacher {i} complete: {epochs_trained}/{epochs} epochs, "
            f"best val acc: {best_acc:.4f}"
        )

    return teachers, teacher_training_info

