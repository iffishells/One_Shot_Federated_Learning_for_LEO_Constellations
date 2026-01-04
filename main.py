#!/usr/bin/env python3
"""
One-Shot Federated Learning for LEO Satellite Constellations
with Data-Free Knowledge Distillation

Paper: "One-Shot Federated Learning for LEO Constellations that Reduces
        Convergence Time from Days to 90 Minutes"

This implementation includes:
- Non-IID data partitioning across satellite orbits
- Teacher model training on each orbit
- Data-free synthetic image generation
- Knowledge distillation to student model
- Comprehensive visualization and results saving

Usage:
    python main.py                          # Run with default config
    python main.py --fast True              # Quick test mode
    python main.py --dataset CIFAR10        # Use CIFAR-10
    python main.py --synthetic False        # Use real data (no generator)
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import from our modular structure
from src.config import Config
from src.models import Generator, make_resnet50, make_resnet18
from src.data import (
    load_dataset, CLASS_NAMES,
    create_orbit_splits, print_orbit_info, subset_by_labels,
    SyntheticDataLoader
)
from src.training import train_teacher, train_generator, train_student_kd
from src.training.teacher import train_all_teachers
from src.training.knowledge_distillation import virtual_retraining
from src.evaluation import (
    eval_acc, eval_ensemble_acc, eval_class_aware_ensemble_acc
)
from src.visualization import generate_all_plots, save_synthetic_images
from src.utils import (
    set_seed, create_experiment_dir, save_results, save_models,
    print_experiment_summary
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="One-Shot Federated Learning with Data-Free Knowledge Distillation"
    )
    parser.add_argument(
        "--dataset", type=str, default="MNIST", choices=["MNIST", "CIFAR10"],
        help="Dataset to use (default: MNIST)"
    )
    parser.add_argument(
        "--fast", type=str, choices=["True", "False"], default="False",
        help="Enable fast mode for quick testing (default: False)"
    )
    parser.add_argument(
        "--synthetic", type=str, choices=["True", "False"], default="True",
        help="Use synthetic data generation (default: True)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="Device to use: 'cuda:0', 'cuda:1', 'cuda' (all GPUs), 'mps', 'cpu', or 'auto' (default: cuda:0)"
    )
    parser.add_argument(
        "--multi-gpu", action="store_true",
        help="Use all available GPUs for training (DataParallel)"
    )
    parser.add_argument(
        "--results-dir", type=str, default="./results",
        help="Directory to save results (default: ./results)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--enable-virtual-retraining", action="store_true",
        help="Enable virtual retraining phase (usually hurts performance)"
    )
    return parser.parse_args()

#
# Example commands:
#   python main.py --dataset MNIST --fast True --synthetic True --device cuda:0
#   python main.py --dataset MNIST --fast True --synthetic True --device cuda:1
#   python main.py --dataset MNIST --fast True --synthetic True --device cuda --multi-gpu  # Use all GPUs
#   python main.py --dataset CIFAR10 --fast False --synthetic True --device cuda --multi-gpu

def main():
    """Main entry point for the experiment."""
    args = parse_args()
    
    # Handle device selection with CUDA and multi-GPU support
    use_multi_gpu = args.multi_gpu or args.device == "cuda"
    gpu_ids = None
    
    if use_multi_gpu and torch.cuda.is_available():
        if torch.cuda.device_count() >= 2:
            device_str = "cuda"
            # Use only GPU 0 and 1
            gpu_ids = [0, 1]
            print(f"✓ Using Multi-GPU mode with GPUs: {gpu_ids}")
        else:
            print(f"⚠️  Only {torch.cuda.device_count()} GPU(s) available. Using single GPU mode.")
            device_str = "cuda:0"
            use_multi_gpu = False
    elif args.device.startswith("cuda"):
        if torch.cuda.is_available():
            # Validate CUDA device index
            device_idx = int(args.device.split(":")[1]) if ":" in args.device else 0
            if device_idx >= torch.cuda.device_count():
                print(f"⚠️  CUDA device {device_idx} not available. Using device 0 instead.")
                device_str = "cuda:0"
            else:
                device_str = args.device
                print(f"✓ Using CUDA device {device_idx}")
    else:
        device_str = args.device
        use_multi_gpu = False
    
    # Create configuration
    config = Config(
        dataset=args.dataset,
        fast_mode=args.fast == "True",
        random_seed=args.seed,
        results_dir=args.results_dir,
        device=device_str,
        use_synthetic_data=args.synthetic == "True",
    )
    config.virtual_retraining.enabled = args.enable_virtual_retraining
    
    # Setup device
    if use_multi_gpu:
        device = torch.device("cuda:0")  # Primary device for DataParallel
        print(f"Primary Device: {device} (Multi-GPU mode enabled with GPUs: {gpu_ids})")
    else:
        device = torch.device(config.device)
        print(f"Device: {device}")
    
    set_seed(config.random_seed)
    
    # Create experiment directory
    experiment_dir, experiment_id, timestamp = create_experiment_dir(
        config.results_dir, config.dataset, config.use_synthetic_data
    )
    
    config.print_config()
    print(f"Experiment ID: {experiment_id}")
    print(f"Results Directory: {experiment_dir}")
    
    # =========================================================================
    # PHASE 0: Data Loading and Partitioning
    # =========================================================================
    print(f"\n{'='*60}")
    print("PHASE 0: Loading Data")
    print(f"{'='*60}")
    
    train_dataset, test_dataset = load_dataset(config.dataset)
    class_names = CLASS_NAMES[config.dataset]
    
    # Create non-IID orbit splits
    print("\nSplitting dataset into orbits...")
    train_subsets, val_subsets = create_orbit_splits(
        train_dataset,
        config.orbit_labels,
        train_val_split=config.train_val_split,
        fast_mode=config.fast_mode
    )
    
    print_orbit_info(config.orbit_labels, train_subsets, val_subsets, class_names)
    
    # =============== ==========================================================
    # PHASE 1: Teacher Training
    # =========================================================================
    print(f"\n{'='*60}")
    print("PHASE 1: Training Teacher Models")
    print(f"{'='*60}")
    
    teachers, teacher_training_info = train_all_teachers(
        teacher_factory=make_resnet50,
        train_subsets=train_subsets,
        val_subsets=val_subsets,
        orbit_labels=config.orbit_labels,
        device=device,
        epochs=config.get_teacher_epochs(),
        lr=config.teacher.learning_rate,
        batch_size=config.teacher.batch_size,
        early_stopping=config.teacher.early_stopping,
        patience=config.teacher.patience,
        min_delta=config.teacher.min_delta,
        use_multi_gpu=use_multi_gpu,
        gpu_ids=gpu_ids if use_multi_gpu else None
    )
    
    # =========================================================================
    # PHASE 2: Data-Free Synthetic Data Generation (Optional)
    # =========================================================================
    generator = None
    
    if config.use_synthetic_data:
        print(f"\n{'='*60}")
        print("PHASE 2: Training Generator (Data-Free Mode)")
        print(f"{'='*60}")
        
        generator = Generator(
            latent_dim=config.generator.latent_dim,
            img_size=224,
            channels=3,
            ngf=64
        )
        
        # Wrap generator with DataParallel if multi-GPU is enabled
        if use_multi_gpu and gpu_ids is not None:
            generator = torch.nn.DataParallel(generator, device_ids=gpu_ids)
            print(f"Wrapped generator with DataParallel for GPUs: {gpu_ids}")
        
        # Infer n_classes from orbit_labels
        all_labels = set()
        for labels in config.orbit_labels:
            all_labels.update(labels)
        n_classes = max(all_labels) + 1 if all_labels else 10
        
        generator = train_generator(
            generator=generator,
            teachers=teachers,
            orbit_labels=config.orbit_labels,
            device=device,
            epochs=config.get_generator_epochs(),
            batch_size=config.get_generator_batch_size(),
            lr=config.generator.learning_rate,
            diversity_weight=config.generator.diversity_weight,
            n_classes=n_classes,
            use_multi_gpu=use_multi_gpu
        )
        
        # Unwrap DataParallel for saving
        if isinstance(generator, torch.nn.DataParallel):
            generator = generator.module
        
        # Create synthetic data loader for KD
        kd_loader = SyntheticDataLoader(
            generator=generator,
            num_batches=config.get_num_kd_batches(),
            batch_size=config.get_kd_batch_size(),
            device=device
        )
        print(f"Synthetic KD loader: {len(kd_loader)} batches of synthetic images")
        
        # Save synthetic images for visualization
        synthetic_dir = os.path.join(experiment_dir, "synthetic_images")
        save_synthetic_images(
            generator, teachers, config.orbit_labels,
            config.dataset, class_names, device, synthetic_dir
        )
    else:
        print("\nUsing real test data for KD (comparison mode)...")
        kd_loader = DataLoader(
            test_dataset,
            batch_size=config.get_kd_batch_size(),
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
    
    # Eval loader always uses real test data
    eval_loader = DataLoader(
        test_dataset,
        batch_size=config.get_kd_batch_size(),
        num_workers=2,
        pin_memory=True
    )
    
    # =========================================================================
    # Evaluate Teachers and Ensembles (Baseline)
    # =========================================================================
    print(f"\n{'='*60}")
    print("Evaluating Teachers and Ensembles")
    print(f"{'='*60}")
    
    # Infer n_classes from orbit_labels
    all_labels = set()
    for labels in config.orbit_labels:
        all_labels.update(labels)
    n_classes = max(all_labels) + 1 if all_labels else 10
    
    teacher_accs = [eval_acc(t, eval_loader, device) 
                   for t in tqdm(teachers, desc="Evaluating teachers")]
    ens_acc = eval_ensemble_acc(teachers, eval_loader, device)
    class_aware_ens_acc = eval_class_aware_ensemble_acc(
        teachers, eval_loader, config.orbit_labels, device, n_classes=n_classes
    )
    
    print(f"\nTeacher accuracies: {[f'{a:.4f}' for a in teacher_accs]}")
    print(f"Naive ensemble accuracy: {ens_acc:.4f}")
    print(f"Class-aware ensemble accuracy: {class_aware_ens_acc:.4f}")
    
    # =========================================================================
    # PHASE 3: Knowledge Distillation
    # =========================================================================
    print(f"\n{'='*60}")
    print("PHASE 3: Knowledge Distillation")
    print(f"{'='*60}")
    
    student = make_resnet18()
    
    # Wrap student with DataParallel if multi-GPU is enabled
    if use_multi_gpu and gpu_ids is not None:
        student = torch.nn.DataParallel(student, device_ids=gpu_ids)
        print(f"Wrapped student with DataParallel for GPUs: {gpu_ids}")
    
    student, final_acc, best_acc, epochs_trained = train_student_kd(
        student=student,
        teachers=teachers,
        kd_loader=kd_loader,
        eval_loader=eval_loader,
        orbit_labels=config.orbit_labels,
        device=device,
        epochs=config.get_kd_epochs(),
        temperature=config.kd.temperature,
        lr=config.kd.learning_rate,
        confidence_threshold=config.kd.confidence_threshold,
        gradient_clip=config.kd.gradient_clip,
        use_class_aware=config.kd.use_class_aware,
        early_stopping=config.kd.early_stopping,
        patience=config.kd.patience,
        min_delta=config.kd.min_delta
    )
    
    # Unwrap DataParallel for evaluation and saving
    if isinstance(student, torch.nn.DataParallel):
        student = student.module
    
    # =========================================================================
    # PHASE 4: Virtual Retraining (Optional)
    # =========================================================================
    virtual_retrain_acc = None
    
    if config.virtual_retraining.enabled:
        print(f"\n{'='*60}")
        print("PHASE 4: Virtual Retraining")
        print(f"{'='*60}")
        
        # Create orbit-style partitions from test data
        parts = [
            subset_by_labels(test_dataset, config.orbit_labels[2], limit=1000, 
                           fast_mode=config.fast_mode),
            subset_by_labels(test_dataset, config.orbit_labels[3], limit=1000,
                           fast_mode=config.fast_mode),
            subset_by_labels(test_dataset, config.orbit_labels[4], limit=1000,
                           fast_mode=config.fast_mode),
        ]
        
        student = virtual_retraining(
            student, parts, device,
            epochs=config.virtual_retraining.epochs,
            lr=config.virtual_retraining.learning_rate
        )
        virtual_retrain_acc = eval_acc(student, eval_loader, device)
        print(f"After virtual retraining, student accuracy: {virtual_retrain_acc:.4f}")
    else:
        print("\nVirtual retraining disabled")
    
    # =========================================================================
    # Save Results
    # =========================================================================
    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print(f"{'='*60}")
    
    # Save results JSON
    save_results(
        experiment_dir=experiment_dir,
        experiment_id=experiment_id,
        timestamp=timestamp,
        config=config,
        teacher_accs=teacher_accs,
        ens_acc=ens_acc,
        class_aware_ens_acc=class_aware_ens_acc,
        student_acc=final_acc,
        best_student_acc=best_acc,
        epochs_trained=epochs_trained,
        teacher_training_info=teacher_training_info,
        train_subsets=train_subsets,
        val_subsets=val_subsets,
        virtual_retrain_acc=virtual_retrain_acc
    )
    
    # Save models
    save_models(
        experiment_dir=experiment_dir,
        student=student,
        teachers=teachers,
        teacher_accs=teacher_accs,
        student_acc=final_acc,
        orbit_labels=config.orbit_labels,
        generator=generator,
        use_synthetic=config.use_synthetic_data
    )
    
    # =========================================================================
    # Generate Plots
    # =========================================================================
    print(f"\n{'='*60}")
    print("GENERATING PLOTS")
    print(f"{'='*60}")
    
    plots_dir = os.path.join(experiment_dir, "plots")
    generate_all_plots(
        teacher_accs=teacher_accs,
        ens_acc=ens_acc,
        class_aware_ens_acc=class_aware_ens_acc,
        student_acc=final_acc,
        orbit_labels=config.orbit_labels,
        class_names=class_names,
        dataset=config.dataset,
        use_synthetic=config.use_synthetic_data,
        experiment_id=experiment_id,
        plots_dir=plots_dir,
        generator=generator,
        teachers=teachers,
        eval_loader=eval_loader,
        device=device
    )
    
    # =========================================================================
    # Print Summary
    # =========================================================================
    print_experiment_summary(
        experiment_id=experiment_id,
        dataset=config.dataset,
        use_synthetic=config.use_synthetic_data,
        teacher_accs=teacher_accs,
        teacher_training_info=teacher_training_info,
        teacher_epochs=config.get_teacher_epochs(),
        ens_acc=ens_acc,
        class_aware_ens_acc=class_aware_ens_acc,
        student_acc=final_acc,
        best_student_acc=best_acc,
        kd_epochs_trained=epochs_trained,
        kd_epochs_total=config.get_kd_epochs(),
        virtual_retrain_acc=virtual_retrain_acc,
        experiment_dir=experiment_dir,
        generator_epochs=config.get_generator_epochs() if config.use_synthetic_data else None,
        num_kd_batches=config.get_num_kd_batches() if config.use_synthetic_data else None
    )
    
    return {
        "experiment_id": experiment_id,
        "student_accuracy": final_acc,
        "best_student_accuracy": best_acc,
        "class_aware_ensemble_accuracy": class_aware_ens_acc,
    }


if __name__ == "__main__":
    main()


