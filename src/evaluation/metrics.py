"""
Evaluation metrics and utilities for knowledge distillation.
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


def eval_acc(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """
    Evaluate model accuracy on a data loader.
    
    Args:
        model: Model to evaluate
        loader: Data loader
        device: Device to run evaluation on
    
    Returns:
        Accuracy as a float between 0 and 1
    """
    model.eval()
    total = 0
    correct = 0
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
    
    return correct / total


def eval_ensemble_acc(
    teachers: List[nn.Module], 
    loader: DataLoader, 
    device: torch.device,
    weights: Optional[List[float]] = None
) -> float:
    """
    Evaluate ensemble accuracy using weighted averaging.
    
    Args:
        teachers: List of teacher models
        loader: Data loader
        device: Device to run evaluation on
        weights: Optional weights for each teacher (uniform if None)
    
    Returns:
        Ensemble accuracy as a float
    """
    for m in teachers:
        m.eval()
    
    total = 0
    correct = 0
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = [m(x) for m in teachers]
            
            if weights is None:
                ens = torch.stack(logits).mean(dim=0)
            else:
                w = torch.tensor(weights, device=device).view(-1, 1, 1)
                ens = (torch.stack(logits) * w).sum(dim=0)
            
            correct += (ens.argmax(1) == y).sum().item()
            total += x.size(0)
    
    return correct / total


def class_aware_ensemble(
    teachers: List[nn.Module], 
    x: torch.Tensor, 
    orbit_labels: List[List[int]],
    device: torch.device,
    n_classes: int = 10,
    temperature: float = 1.0,
    return_logits: bool = False
) -> torch.Tensor:
    """
    Class-aware ensemble where each teacher contributes only to classes it was trained on.
    
    This is the key innovation for handling non-IID data across satellite orbits.
    Each teacher only votes on classes it has seen during training.
    
    Args:
        teachers: List of teacher models
        x: Input tensor of shape (batch_size, C, H, W)
        orbit_labels: List of class labels each teacher was trained on
        device: Device for computation
        n_classes: Total number of classes (inferred from orbit_labels if not provided)
        temperature: Temperature for softmax scaling (default: 1.0)
        return_logits: If True, return logits instead of probabilities
    
    Returns:
        Ensemble probabilities (or logits if return_logits=True) of shape (batch_size, n_classes)
    """
    batch_size = x.size(0)
    
    # Infer n_classes from orbit_labels if not provided
    if n_classes is None:
        all_labels = set()
        for labels in orbit_labels:
            all_labels.update(labels)
        n_classes = max(all_labels) + 1 if all_labels else 10
    
    # Accumulate weighted votes per class (using logits for temperature scaling)
    vote_sum = torch.zeros(batch_size, n_classes, device=device)
    vote_count = torch.zeros(n_classes, device=device)

    for teacher, labels in zip(teachers, orbit_labels):
        logits = teacher(x)
        # Apply temperature to logits before accumulating
        logits_scaled = logits / temperature
        for c in labels:
            vote_sum[:, c] += logits_scaled[:, c]
            vote_count[c] += 1

    # Average logits per class
    vote_count = vote_count.clamp(min=1)
    ensemble_logits = vote_sum / vote_count.unsqueeze(0)
    
    if return_logits:
        return ensemble_logits
    
    # Convert to probabilities
    ensemble_probs = F.softmax(ensemble_logits, dim=1)
    return ensemble_probs


def eval_class_aware_ensemble_acc(
    teachers: List[nn.Module], 
    loader: DataLoader, 
    orbit_labels: List[List[int]],
    device: torch.device,
    n_classes: int = 10
) -> float:
    """
    Evaluate class-aware ensemble accuracy.
    
    Args:
        teachers: List of teacher models
        loader: Data loader
        orbit_labels: Class labels each teacher was trained on
        device: Device for computation
        n_classes: Total number of classes (inferred if not provided)
    
    Returns:
        Class-aware ensemble accuracy
    """
    for m in teachers:
        m.eval()
    
    total = 0
    correct = 0
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            ens_probs = class_aware_ensemble(teachers, x, orbit_labels, device, n_classes=n_classes)
            correct += (ens_probs.argmax(1) == y).sum().item()
            total += x.size(0)
    
    return correct / total


def evaluate_all_models(
    teachers: List[nn.Module],
    student: nn.Module,
    eval_loader: DataLoader,
    orbit_labels: List[List[int]],
    device: torch.device,
    n_classes: int = None
) -> dict:
    """
    Evaluate all models and return comprehensive metrics.
    
    Args:
        teachers: List of teacher models
        student: Student model
        eval_loader: Evaluation data loader
        orbit_labels: Class labels per orbit
        device: Device for computation
        n_classes: Total number of classes (inferred if not provided)
    
    Returns:
        Dictionary with all evaluation metrics
    """
    # Infer n_classes from orbit_labels if not provided
    if n_classes is None:
        all_labels = set()
        for labels in orbit_labels:
            all_labels.update(labels)
        n_classes = max(all_labels) + 1 if all_labels else 10
    
    print("Evaluating all models...")
    
    # Evaluate individual teachers
    teacher_accs = [
        eval_acc(t, eval_loader, device) 
        for t in tqdm(teachers, desc="Evaluating teachers")
    ]
    
    # Evaluate ensembles
    naive_ens_acc = eval_ensemble_acc(teachers, eval_loader, device)
    class_aware_ens_acc = eval_class_aware_ensemble_acc(
        teachers, eval_loader, orbit_labels, device, n_classes=n_classes
    )
    
    # Evaluate student
    student_acc = eval_acc(student, eval_loader, device)
    
    metrics = {
        "teacher_accuracies": teacher_accs,
        "best_teacher_accuracy": max(teacher_accs),
        "naive_ensemble_accuracy": naive_ens_acc,
        "class_aware_ensemble_accuracy": class_aware_ens_acc,
        "student_accuracy": student_acc,
    }
    
    print(f"\nTeacher accuracies: {[f'{a:.4f}' for a in teacher_accs]}")
    print(f"Naive ensemble accuracy: {naive_ens_acc:.4f}")
    print(f"Class-aware ensemble accuracy: {class_aware_ens_acc:.4f}")
    print(f"Student accuracy: {student_acc:.4f}")
    
    return metrics

