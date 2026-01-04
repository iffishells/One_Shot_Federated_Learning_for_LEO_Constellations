"""
Utility functions for the knowledge distillation pipeline.
"""

import os
import json
import random
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_experiment_dir(
    base_dir: str,
    dataset: str,
    use_synthetic: bool
) -> tuple:
    """
    Create experiment directory with timestamp.
    
    Args:
        base_dir: Base results directory
        dataset: Dataset name
        use_synthetic: Whether using synthetic data
    
    Returns:
        Tuple of (experiment_dir, experiment_id, timestamp)
    """
    os.makedirs(base_dir, exist_ok=True)
    
    data_mode = "synthetic_data" if use_synthetic else "without_synthetic_data"
    experiment_name = f"KD_{dataset}_ClassAware_{data_mode}"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"{experiment_name}_{timestamp}"
    experiment_dir = os.path.join(base_dir, experiment_id)
    os.makedirs(experiment_dir, exist_ok=True)
    
    return experiment_dir, experiment_id, timestamp


def save_results(
    experiment_dir: str,
    experiment_id: str,
    timestamp: str,
    config,  # Config object
    teacher_accs: List[float],
    ens_acc: float,
    class_aware_ens_acc: float,
    student_acc: float,
    best_student_acc: float,
    epochs_trained: int,
    teacher_training_info: List[Dict],
    train_subsets: List,
    val_subsets: List,
    virtual_retrain_acc: Optional[float] = None
) -> str:
    """
    Save experiment results to JSON file.
    
    Returns:
        Path to the saved results file
    """
    results = {
        "experiment_id": experiment_id,
        "timestamp": timestamp,
        "config": config.to_dict(),
        "metrics": {
            "teacher_accuracies": teacher_accs,
            "naive_ensemble_accuracy": ens_acc,
            "class_aware_ensemble_accuracy": class_aware_ens_acc,
            "final_student_accuracy": student_acc,
            "best_student_accuracy": best_student_acc,
            "epochs_trained": epochs_trained,
            "early_stopped": epochs_trained < config.get_kd_epochs(),
            "virtual_retrain_accuracy": virtual_retrain_acc,
        },
        "orbit_splits": {
            f"orbit_{i + 1}": {
                "labels": config.orbit_labels[i],
                "train_size": len(train_subsets[i]),
                "val_size": len(val_subsets[i]),
            }
            for i in range(len(config.orbit_labels))
        },
    }

    results_path = os.path.join(experiment_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_path}")
    return results_path


def save_models(
    experiment_dir: str,
    student,
    teachers: List,
    teacher_accs: List[float],
    student_acc: float,
    orbit_labels: List[List[int]],
    generator=None,
    use_synthetic: bool = False
):
    """
    Save all trained models.
    
    Args:
        experiment_dir: Directory to save models
        student: Trained student model
        teachers: List of trained teacher models
        teacher_accs: Teacher accuracies
        student_acc: Student accuracy
        orbit_labels: Class labels per orbit
        generator: Generator model (if synthetic data used)
        use_synthetic: Whether synthetic data was used
    """
    # Save student model
    student_path = os.path.join(experiment_dir, "student_model.pt")
    torch.save({
        "model_state_dict": student.state_dict(),
        "accuracy": student_acc,
    }, student_path)
    print(f"Student model saved to: {student_path}")

    # Save generator model (if applicable)
    if use_synthetic and generator is not None:
        generator_path = os.path.join(experiment_dir, "generator_model.pt")
        # Get actual generator module (unwrap DataParallel if needed)
        actual_generator = generator.module if isinstance(generator, torch.nn.DataParallel) else generator
        torch.save({
            "model_state_dict": actual_generator.state_dict(),
            "latent_dim": actual_generator.latent_dim,
            "img_size": actual_generator.img_size,
        }, generator_path)
        print(f"Generator model saved to: {generator_path}")

    # Save teacher models
    teachers_dir = os.path.join(experiment_dir, "teachers")
    os.makedirs(teachers_dir, exist_ok=True)
    
    for i, (teacher, acc) in enumerate(zip(teachers, teacher_accs)):
        teacher_path = os.path.join(teachers_dir, f"teacher_orbit{i + 1}.pt")
        torch.save({
            "model_state_dict": teacher.state_dict(),
            "orbit_labels": orbit_labels[i],
            "accuracy": acc,
        }, teacher_path)
    
    print(f"Teacher models saved to: {teachers_dir}/")


def avg_state_dicts(dicts: List[Dict]) -> Dict:
    """
    Average multiple state dictionaries (for FedAvg-style aggregation).
    
    Args:
        dicts: List of state dictionaries
    
    Returns:
        Averaged state dictionary
    """
    avg = {}
    for k in dicts[0].keys():
        avg[k] = sum(d[k] for d in dicts) / len(dicts)
    return avg


def print_experiment_summary(
    experiment_id: str,
    dataset: str,
    use_synthetic: bool,
    teacher_accs: List[float],
    teacher_training_info: List[Dict],
    teacher_epochs: int,
    ens_acc: float,
    class_aware_ens_acc: float,
    student_acc: float,
    best_student_acc: float,
    kd_epochs_trained: int,
    kd_epochs_total: int,
    virtual_retrain_acc: Optional[float],
    experiment_dir: str,
    generator_epochs: Optional[int] = None,
    num_kd_batches: Optional[int] = None
):
    """
    Print comprehensive experiment summary.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Experiment ID: {experiment_id}")
    print(f"Dataset: {dataset}")
    print(f"Data-Free Mode: {'YES (synthetic data)' if use_synthetic else 'NO (real test data)'}")
    
    if use_synthetic:
        print(f"  Generator epochs: {generator_epochs}")
        print(f"  KD batches: {num_kd_batches}")
    
    print(f"\nTeacher accuracies: {[f'{a:.4f}' for a in teacher_accs]}")
    
    for i, info in enumerate(teacher_training_info):
        status = " (early stopped)" if info["early_stopped"] else ""
        print(f"  Teacher {i + 1}: {info['epochs_trained']}/{teacher_epochs} epochs, "
              f"best val acc: {info['best_val_acc']:.4f}{status}")
    
    print(f"\nNaive ensemble accuracy: {ens_acc:.4f}")
    print(f"Class-aware ensemble accuracy: {class_aware_ens_acc:.4f}")
    print(f"Final student accuracy: {student_acc:.4f}")
    print(f"Best student accuracy: {best_student_acc:.4f}")
    print(f"Epochs trained: {kd_epochs_trained}/{kd_epochs_total}" + 
          (" (early stopped)" if kd_epochs_trained < kd_epochs_total else ""))
    
    if virtual_retrain_acc is not None:
        print(f"After virtual retraining: {virtual_retrain_acc:.4f}")
    
    print(f"\nAll results saved to: {experiment_dir}")
    print("=" * 60)

