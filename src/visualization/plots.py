"""
Visualization and plotting utilities for experiment results.
"""

import os
from typing import List, Optional
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Patch

matplotlib.use('Agg')  # Non-interactive backend for saving plots


def plot_accuracy_comparison(
    teacher_accs: List[float],
    ens_acc: float,
    class_aware_ens_acc: float,
    student_acc: float,
    dataset: str,
    use_synthetic: bool,
    save_path: str
):
    """
    Plot model accuracy comparison bar chart.
    
    Args:
        teacher_accs: List of teacher accuracies
        ens_acc: Naive ensemble accuracy
        class_aware_ens_acc: Class-aware ensemble accuracy
        student_acc: Student accuracy
        dataset: Dataset name
        use_synthetic: Whether synthetic data was used
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    models = [f'Teacher {i + 1}\n(Orbit {i + 1})' for i in range(len(teacher_accs))]
    models += ['Naive\nEnsemble', 'Class-Aware\nEnsemble', 'Student']
    accuracies = teacher_accs + [ens_acc, class_aware_ens_acc, student_acc]
    colors = ['#3498db'] * len(teacher_accs) + ['#e74c3c', '#2ecc71', '#9b59b6']

    bars = ax.bar(models, [a * 100 for a in accuracies], color=colors, 
                  edgecolor='black', linewidth=1.2)

    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.annotate(f'{acc * 100:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_xlabel('Model', fontsize=14)
    ax.set_title(f'{dataset} - Model Accuracy Comparison\n'
                 f'({"Data-Free" if use_synthetic else "With Real Data"})',
                 fontsize=16, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)

    legend_elements = [
        Patch(facecolor='#3498db', label='Individual Teachers'),
        Patch(facecolor='#e74c3c', label='Naive Ensemble'),
        Patch(facecolor='#2ecc71', label='Class-Aware Ensemble'),
        Patch(facecolor='#9b59b6', label='Distilled Student'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_orbit_distribution(
    orbit_labels: List[List[int]],
    class_names: List[str],
    dataset: str,
    save_path: str
):
    """
    Plot orbit class distribution heatmap.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    orbit_matrix = np.zeros((len(orbit_labels), 10))
    for i, labels in enumerate(orbit_labels):
        for l in labels:
            orbit_matrix[i, l] = 1

    im = ax.imshow(orbit_matrix, cmap='Blues', aspect='auto')

    ax.set_xticks(range(10))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticks(range(len(orbit_labels)))
    ax.set_yticklabels([f'Orbit {i + 1}\n({len(orbit_labels[i])} classes)' 
                        for i in range(len(orbit_labels))])

    ax.set_xlabel('Classes', fontsize=14)
    ax.set_ylabel('Orbits', fontsize=14)
    ax.set_title(f'{dataset} - Non-IID Data Distribution Across Orbits', 
                 fontsize=16, fontweight='bold')

    for i in range(len(orbit_labels)):
        for j in range(10):
            text = '✓' if orbit_matrix[i, j] == 1 else ''
            ax.text(j, i, text, ha='center', va='center', fontsize=14)

    plt.colorbar(im, ax=ax, label='Has Class', shrink=0.8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_teacher_analysis(
    teacher_accs: List[float],
    orbit_labels: List[List[int]],
    save_path: str
):
    """
    Plot teacher accuracies with orbit information.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(teacher_accs))
    bars = ax1.bar(x, [a * 100 for a in teacher_accs], color='#3498db', edgecolor='black')

    for bar, acc in zip(bars, teacher_accs):
        height = bar.get_height()
        ax1.annotate(f'{acc * 100:.1f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax1.set_xlabel('Teacher (Orbit)', fontsize=12)
    ax1.set_ylabel('Accuracy on Full Test Set (%)', fontsize=12)
    ax1.set_title('Individual Teacher Accuracies\n(Evaluated on ALL 10 classes)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'T{i + 1}\n{len(orbit_labels[i])} cls' 
                         for i in range(len(teacher_accs))])
    ax1.set_ylim(0, 100)

    num_classes = [len(labels) for labels in orbit_labels]
    ax2.bar(x, num_classes, color='#2ecc71', edgecolor='black')
    ax2.set_xlabel('Orbit', fontsize=12)
    ax2.set_ylabel('Number of Classes', fontsize=12)
    ax2.set_title('Non-IID Distribution\n(Classes per Orbit)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Orbit {i + 1}' for i in range(len(orbit_labels))])
    ax2.set_ylim(0, 10)
    ax2.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='All classes')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_aggregation_comparison(
    teacher_accs: List[float],
    ens_acc: float,
    class_aware_ens_acc: float,
    student_acc: float,
    dataset: str,
    use_synthetic: bool,
    save_path: str
):
    """
    Plot horizontal bar chart comparing aggregation methods.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = ['Best Single\nTeacher', 'Naive\nEnsemble', 
               'Class-Aware\nEnsemble', 'Distilled\nStudent']
    accs = [max(teacher_accs), ens_acc, class_aware_ens_acc, student_acc]
    colors = ['#95a5a6', '#e74c3c', '#2ecc71', '#9b59b6']

    bars = ax.barh(methods, [a * 100 for a in accs], color=colors, 
                   edgecolor='black', height=0.6)

    for bar, acc in zip(bars, accs):
        width = bar.get_width()
        ax.annotate(f'{acc * 100:.1f}%',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha='left', va='center', fontsize=12, fontweight='bold')

    ax.set_xlabel('Accuracy (%)', fontsize=14)
    ax.set_title(f'{dataset} - Knowledge Aggregation Methods\n'
                 f'({"Data-Free Mode" if use_synthetic else "With Real Data"})',
                 fontsize=16, fontweight='bold')
    ax.set_xlim(0, 110)
    ax.axvline(x=100, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_real_vs_synthetic(
    generator,
    teachers: List,
    orbit_labels: List[List[int]],
    eval_loader,
    dataset: str,
    class_names: List[str],
    device: torch.device,
    save_path: str
):
    """
    Plot comparison of real vs synthetic images.
    """
    fig, axes = plt.subplots(2, 8, figsize=(20, 6))
    fig.suptitle(f'{dataset} - Real vs Synthetic Images Comparison', 
                 fontsize=16, fontweight='bold')

    # Get real images
    real_iter = iter(eval_loader)
    real_batch, real_labels = next(real_iter)

    # Denormalize
    if dataset == "MNIST":
        mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    else:
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)

    real_denorm = real_batch[:8] * std + mean
    real_denorm = real_denorm.clamp(0, 1)

    # Generate synthetic images
    generator.eval()
    # Get actual generator module (unwrap DataParallel if needed)
    actual_generator = generator.module if isinstance(generator, torch.nn.DataParallel) else generator
    with torch.no_grad():
        z = torch.randn(8, actual_generator.latent_dim, device=device)
        synth_batch = generator(z)

        # Get predictions
        vote_sum = torch.zeros(8, 10, device=device)
        vote_count = torch.zeros(10, device=device)
        for teacher, labels in zip(teachers, orbit_labels):
            logits = teacher(synth_batch)
            probs = F.softmax(logits, dim=1)
            for c in labels:
                vote_sum[:, c] += probs[:, c]
                vote_count[c] += 1
        vote_count = vote_count.clamp(min=1)
        synth_probs = vote_sum / vote_count.unsqueeze(0)
        synth_probs = synth_probs / (synth_probs.sum(dim=1, keepdim=True) + 1e-8)
        synth_preds = synth_probs.argmax(dim=1)
        synth_confs = synth_probs.max(dim=1)[0]

    synth_denorm = synth_batch.cpu() * std + mean
    synth_denorm = synth_denorm.clamp(0, 1)

    # Plot real images
    for i in range(8):
        ax = axes[0, i]
        img = real_denorm[i].permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_title(f'Real: {class_names[real_labels[i].item()]}', fontsize=10)
        ax.axis('off')
        if i == 0:
            ax.set_ylabel('REAL', fontsize=12, fontweight='bold')

    # Plot synthetic images
    for i in range(8):
        ax = axes[1, i]
        img = synth_denorm[i].permute(1, 2, 0).numpy()
        ax.imshow(img)
        pred_class = class_names[synth_preds[i].item()]
        conf = synth_confs[i].item()
        ax.set_title(f'Synth: {pred_class}\n({conf * 100:.0f}%)', fontsize=10)
        ax.axis('off')
        if i == 0:
            ax.set_ylabel('SYNTHETIC', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_summary_dashboard(
    teacher_accs: List[float],
    ens_acc: float,
    class_aware_ens_acc: float,
    student_acc: float,
    orbit_labels: List[List[int]],
    dataset: str,
    use_synthetic: bool,
    experiment_id: str,
    save_path: str
):
    """
    Generate a comprehensive summary dashboard.
    """
    fig = plt.figure(figsize=(16, 10))

    fig.suptitle(f'{dataset} One-Shot Federated Learning Results\n{experiment_id}',
                 fontsize=18, fontweight='bold', y=0.98)

    # Subplot 1: Accuracy bars
    ax1 = fig.add_subplot(2, 2, 1)
    models_short = [f'T{i + 1}' for i in range(len(teacher_accs))] + ['Naive', 'ClassAware', 'Student']
    all_accs = teacher_accs + [ens_acc, class_aware_ens_acc, student_acc]
    colors = ['#3498db'] * len(teacher_accs) + ['#e74c3c', '#2ecc71', '#9b59b6']
    ax1.bar(models_short, [a * 100 for a in all_accs], color=colors, edgecolor='black')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Model Accuracies')
    ax1.set_ylim(0, 105)
    for i, acc in enumerate(all_accs):
        ax1.text(i, acc * 100 + 2, f'{acc * 100:.1f}', ha='center', fontsize=9)

    # Subplot 2: Orbit heatmap
    ax2 = fig.add_subplot(2, 2, 2)
    orbit_matrix = np.zeros((len(orbit_labels), 10))
    for i, labels in enumerate(orbit_labels):
        for l in labels:
            orbit_matrix[i, l] = 1
    im = ax2.imshow(orbit_matrix, cmap='Blues', aspect='auto')
    ax2.set_xticks(range(10))
    ax2.set_xticklabels([str(i) for i in range(10)])
    ax2.set_yticks(range(len(orbit_labels)))
    ax2.set_yticklabels([f'O{i + 1}' for i in range(len(orbit_labels))])
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Orbit')
    ax2.set_title('Non-IID Distribution')

    # Subplot 3: Key metrics
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.axis('off')
    metrics_text = f"""
EXPERIMENT CONFIGURATION
━━━━━━━━━━━━━━━━━━━━━━━━━━
Dataset: {dataset}
Data-Free Mode: {'Yes' if use_synthetic else 'No'}
Number of Orbits: {len(orbit_labels)}
Teacher Model: ResNet-50
Student Model: ResNet-18

KEY RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━
Best Teacher: {max(teacher_accs) * 100:.1f}%
Naive Ensemble: {ens_acc * 100:.1f}%
Class-Aware Ensemble: {class_aware_ens_acc * 100:.1f}%
Student (Final): {student_acc * 100:.1f}%

IMPROVEMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━
Ensemble vs Best Teacher: +{(class_aware_ens_acc - max(teacher_accs)) * 100:.1f}%
Student vs Best Teacher: {'+' if student_acc > max(teacher_accs) else ''}{(student_acc - max(teacher_accs)) * 100:.1f}%
"""
    ax3.text(0.1, 0.95, metrics_text, transform=ax3.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Subplot 4: Comparison chart
    ax4 = fig.add_subplot(2, 2, 4)
    comparison = ['Best Teacher', 'Class-Aware Ens.', 'Student']
    comp_accs = [max(teacher_accs) * 100, class_aware_ens_acc * 100, student_acc * 100]
    comp_colors = ['#3498db', '#2ecc71', '#9b59b6']
    wedges, texts, autotexts = ax4.pie(comp_accs, labels=comparison, colors=comp_colors,
                                       autopct='%1.1f%%', startangle=90,
                                       explode=(0, 0.05, 0.1))
    ax4.set_title('Accuracy Distribution')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def save_synthetic_images(
    generator,
    teachers: List,
    orbit_labels: List[List[int]],
    dataset: str,
    class_names: List[str],
    device: torch.device,
    save_dir: str,
    num_images: int = 64
):
    """
    Generate and save synthetic images with predictions.
    """
    from torchvision.utils import save_image
    
    os.makedirs(save_dir, exist_ok=True)
    
    generator.eval()
    with torch.no_grad():
        # Get actual generator module (unwrap DataParallel if needed)
        actual_generator = generator.module if isinstance(generator, torch.nn.DataParallel) else generator
        z = torch.randn(num_images, actual_generator.latent_dim, device=device)
        synthetic_images = generator(z)

        # Get teacher predictions
        vote_sum = torch.zeros(num_images, 10, device=device)
        vote_count = torch.zeros(10, device=device)

        for teacher, labels in zip(teachers, orbit_labels):
            logits = teacher(synthetic_images)
            probs = F.softmax(logits, dim=1)
            for c in labels:
                vote_sum[:, c] += probs[:, c]
                vote_count[c] += 1

        vote_count = vote_count.clamp(min=1)
        ensemble_probs = vote_sum / vote_count.unsqueeze(0)
        ensemble_probs = ensemble_probs / (ensemble_probs.sum(dim=1, keepdim=True) + 1e-8)

        predicted_classes = ensemble_probs.argmax(dim=1)
        confidences = ensemble_probs.max(dim=1)[0]

    # Denormalize
    if dataset == "MNIST":
        mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
    else:
        mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616], device=device).view(1, 3, 1, 1)

    denorm_images = synthetic_images * std + mean
    denorm_images = denorm_images.clamp(0, 1)

    # Save individual images
    for idx in range(min(32, len(denorm_images))):
        img_path = os.path.join(
            save_dir,
            f"synthetic_{idx:03d}_class{predicted_classes[idx].item()}_"
            f"conf{confidences[idx].item():.2f}.png"
        )
        save_image(denorm_images[idx], img_path)

    # Create grid
    fig, axes = plt.subplots(8, 8, figsize=(16, 16))
    fig.suptitle(f'Synthetic Images Generated by Data-Free KD\n{dataset} Dataset', 
                 fontsize=16, fontweight='bold')

    for idx, ax in enumerate(axes.flat):
        if idx < len(denorm_images):
            img = denorm_images[idx].cpu().permute(1, 2, 0).numpy()
            ax.imshow(img)
            pred_class = predicted_classes[idx].item()
            conf = confidences[idx].item()
            class_name = class_names[pred_class]
            ax.set_title(f'{class_name}\n({conf * 100:.0f}%)', fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    grid_path = os.path.join(save_dir, "synthetic_grid.png")
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved synthetic image grid: {grid_path}")

    # Save batch tensor
    torch.save({
        'images': synthetic_images.cpu(),
        'predicted_classes': predicted_classes.cpu(),
        'confidences': confidences.cpu(),
        'ensemble_probs': ensemble_probs.cpu(),
    }, os.path.join(save_dir, "synthetic_batch.pt"))


def generate_all_plots(
    teacher_accs: List[float],
    ens_acc: float,
    class_aware_ens_acc: float,
    student_acc: float,
    orbit_labels: List[List[int]],
    class_names: List[str],
    dataset: str,
    use_synthetic: bool,
    experiment_id: str,
    plots_dir: str,
    generator=None,
    teachers: List = None,
    eval_loader=None,
    device=None
):
    """
    Generate all plots and save them to the plots directory.
    """
    os.makedirs(plots_dir, exist_ok=True)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12

    # Plot 1: Accuracy comparison
    plot_accuracy_comparison(
        teacher_accs, ens_acc, class_aware_ens_acc, student_acc,
        dataset, use_synthetic,
        os.path.join(plots_dir, "accuracy_comparison.png")
    )

    # Plot 2: Orbit distribution
    plot_orbit_distribution(
        orbit_labels, class_names, dataset,
        os.path.join(plots_dir, "orbit_distribution.png")
    )

    # Plot 3: Teacher analysis
    plot_teacher_analysis(
        teacher_accs, orbit_labels,
        os.path.join(plots_dir, "teacher_analysis.png")
    )

    # Plot 4: Aggregation comparison
    plot_aggregation_comparison(
        teacher_accs, ens_acc, class_aware_ens_acc, student_acc,
        dataset, use_synthetic,
        os.path.join(plots_dir, "aggregation_comparison.png")
    )

    # Plot 5: Real vs Synthetic (if applicable)
    if use_synthetic and generator is not None and teachers is not None:
        plot_real_vs_synthetic(
            generator, teachers, orbit_labels, eval_loader,
            dataset, class_names, device,
            os.path.join(plots_dir, "real_vs_synthetic.png")
        )

    # Plot 6: Summary dashboard
    plot_summary_dashboard(
        teacher_accs, ens_acc, class_aware_ens_acc, student_acc,
        orbit_labels, dataset, use_synthetic, experiment_id,
        os.path.join(plots_dir, "summary_dashboard.png")
    )

    print(f"\nAll plots saved to: {plots_dir}/")

