"""
Knowledge distillation training utilities.
"""

import copy
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..evaluation.metrics import class_aware_ensemble, eval_acc


def train_student_kd(
    student: nn.Module,
    teachers: List[nn.Module],
    kd_loader,  # DataLoader or SyntheticDataLoader
    eval_loader: DataLoader,
    orbit_labels: List[List[int]],
    device: torch.device,
    epochs: int = 20,
    temperature: float = 4.0,
    lr: float = 1e-3,
    confidence_threshold: float = 0.35,
    gradient_clip: float = 1.0,
    use_class_aware: bool = True,
    early_stopping: bool = True,
    patience: int = 5,
    min_delta: float = 0.001
) -> Tuple[nn.Module, float, float, int]:
    """
    Train student model using knowledge distillation from teacher ensemble.
    
    Uses the KL divergence objective: R_KL^S = KL(D_teacher, D_student)
    
    Args:
        student: Student model to train
        teachers: List of trained teacher models
        kd_loader: Data loader for KD training (real or synthetic)
        eval_loader: Data loader for evaluation (always real data)
        orbit_labels: Class labels each teacher was trained on
        device: Device to train on
        epochs: Maximum number of epochs
        temperature: Softmax temperature (higher = softer targets)
        lr: Learning rate
        confidence_threshold: Only learn from predictions above this confidence
        gradient_clip: Gradient clipping value
        use_class_aware: Whether to use class-aware ensemble
        early_stopping: Whether to use early stopping
        patience: Early stopping patience
        min_delta: Minimum improvement for patience reset
    
    Returns:
        Tuple of (trained_student, final_accuracy, best_accuracy, epochs_trained)
    """
    student = student.to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    
    # Put teachers in eval mode
    for teacher in teachers:
        teacher.eval()
    
    # Early stopping setup
    best_acc = 0.0
    best_state = None
    patience_counter = 0
    epochs_trained = 0

    print("\nStarting Knowledge Distillation...")
    if early_stopping:
        print(f"Early stopping enabled: patience={patience}, min_delta={min_delta}")

    kd_epoch_pbar = tqdm(range(epochs), desc="KD Training")
    for ep in kd_epoch_pbar:
        student.train()
        loss_sum = 0
        kept = 0
        
        kd_batch_pbar = tqdm(
            kd_loader, desc=f"KD Epoch {ep + 1}/{epochs}", leave=False
        )
        
        for x, _ in kd_batch_pbar:
            x = x.to(device)
            
            with torch.no_grad():
                if use_class_aware:
                    # Class-aware ensemble: each teacher contributes only to its classes
                    p = class_aware_ensemble(teachers, x, orbit_labels, device)
                else:
                    # Naive average
                    logits_list = [m(x) for m in teachers]
                    D_teacher = torch.stack(logits_list).mean(dim=0)
                    p = F.softmax(D_teacher / temperature, dim=1)

            D_student = student(x)
            q_log = F.log_softmax(D_student / temperature, dim=1)

            # Confidence filter: only learn from confident predictions
            maxp, _ = p.max(dim=1)
            mask = (maxp >= confidence_threshold)
            if mask.sum() == 0:
                continue
            
            kept += mask.sum().item()
            p_sel = p[mask]
            q_log_sel = q_log[mask]

            loss = F.kl_div(q_log_sel, p_sel, reduction='batchmean') * (temperature ** 2)
            
            optimizer.zero_grad()
            loss.backward()
            if gradient_clip:
                nn.utils.clip_grad_norm_(student.parameters(), gradient_clip)
            optimizer.step()
            
            loss_sum += loss.item()
            kd_batch_pbar.set_postfix(loss=f"{loss.item():.4f}", kept=kept)

        # Evaluation
        current_acc = eval_acc(student, eval_loader, device)
        epochs_trained = ep + 1
        
        kd_epoch_pbar.set_postfix(
            acc=f"{current_acc:.3f}", 
            kept=kept, 
            best=f"{best_acc:.3f}"
        )
        tqdm.write(
            f"KD Epoch {ep + 1}: train KL {loss_sum / max(1, kept):.4f}, "
            f"eval acc {current_acc:.3f}, kept {kept}"
        )

        # Early stopping check
        if early_stopping:
            if current_acc > best_acc + min_delta:
                best_acc = current_acc
                best_state = copy.deepcopy(student.state_dict())
                patience_counter = 0
                tqdm.write(f"  ✓ New best accuracy: {best_acc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    tqdm.write(
                        f"\n⚠ Early stopping triggered! "
                        f"No improvement for {patience} epochs."
                    )
                    tqdm.write(f"  Best accuracy: {best_acc:.4f}")
                    if best_state is not None:
                        student.load_state_dict(best_state)
                    break
        else:
            # Just track best without early stopping
            if current_acc > best_acc:
                best_acc = current_acc
                best_state = copy.deepcopy(student.state_dict())

    # Ensure we use the best model
    if best_state is not None and not early_stopping:
        student.load_state_dict(best_state)
        print(f"Loaded best model (acc: {best_acc:.4f})")

    final_acc = eval_acc(student, eval_loader, device)
    print(f"Final student accuracy: {final_acc:.4f} (best during training: {best_acc:.4f})")

    return student.eval(), final_acc, best_acc, epochs_trained


def train_virtual(
    init_model: nn.Module,
    dataset,
    device: torch.device,
    epochs: int = 3,
    lr: float = 5e-4,
    batch_size: int = 64,
    num_workers: int = 2
) -> nn.Module:
    """
    Train a virtual student model for optional retraining phase.
    
    Args:
        init_model: Initial model (typically the distilled student)
        dataset: Dataset to train on
        device: Device to train on
        epochs: Number of epochs
        lr: Learning rate
        batch_size: Batch size
        num_workers: Number of data loader workers
    
    Returns:
        Trained model
    """
    model = copy.deepcopy(init_model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    for ep in range(epochs):
        model.train()
        for x, y in tqdm(loader, desc=f"Virtual ep {ep + 1}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return model.eval()


def virtual_retraining(
    student: nn.Module,
    parts: List,
    device: torch.device,
    epochs: int = 3,
    lr: float = 5e-4,
    batch_size: int = 64
) -> Tuple[nn.Module, float]:
    """
    Perform virtual retraining (Phase 3 from the paper).
    
    Clone several virtual students, train each on an "orbit-style" partition,
    then average their weights.
    
    Note: This often hurts performance because training on non-IID partitions
    causes the student to forget classes.
    
    Args:
        student: Distilled student model
        parts: List of data partitions (mimicking orbit-style splits)
        device: Device to train on
        epochs: Training epochs per virtual student
        lr: Learning rate
        batch_size: Batch size
    
    Returns:
        Tuple of (retrained_student, new_accuracy)
    """
    from ..utils.helpers import avg_state_dicts
    
    print("\nStarting virtual retraining on student...")
    virtual_states = []
    
    for ds in tqdm(parts, desc="Virtual Retraining"):
        vm = train_virtual(student, ds, device, epochs=epochs, lr=lr, batch_size=batch_size)
        virtual_states.append(vm.state_dict())
    
    student.load_state_dict(avg_state_dicts(virtual_states))
    return student

