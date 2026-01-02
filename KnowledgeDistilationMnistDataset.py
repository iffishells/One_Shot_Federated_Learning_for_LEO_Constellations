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
"""

import random, copy, time, os, json
from dataclasses import dataclass, asdict
from typing import List
from datetime import datetime
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # Non-interactive backend for saving plots

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms, models

# ============================================================================
# ========================= CONFIGURATION PARAMETERS =========================
# ============================================================================
# All hyperparameters and settings are defined here for easy modification.
# ============================================================================

if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # GENERAL SETTINGS
    # -------------------------------------------------------------------------
    FAST = False  # True: Quick test mode (fewer epochs)
    # False: Full training (paper settings)

    DATASET = "MNIST"  # Options: "MNIST" or "CIFAR10"

    RANDOM_SEED = 42  # Random seed for reproducibility

    RESULTS_DIR = "./results"  # Directory to save all results

    # -------------------------------------------------------------------------
    # ORBIT / FEDERATED LEARNING SETTINGS
    # -------------------------------------------------------------------------
    # Non-IID data distribution: each orbit sees only a subset of classes
    ORBIT_LABELS = [
        [0, 1, 2, 3],  # Orbit 1: 4 classes
        [4, 5, 6, 7],  # Orbit 2: 4 classes
        [0, 1, 2, 3, 4, 5],  # Orbit 3: 6 classes
        [6, 7, 8, 9, 0, 1],  # Orbit 4: 6 classes
        [2, 3, 4, 5, 8, 9],  # Orbit 5: 6 classes
    ]

    TRAIN_VAL_SPLIT = 0.85  # 85% train, 15% validation per orbit

    # -------------------------------------------------------------------------
    # TEACHER MODEL SETTINGS (Paper: I=300, lr=0.001, bs=32)
    # -------------------------------------------------------------------------
    TEACHER_MODEL = "ResNet50"  # Teacher architecture

    TEACHER_EPOCHS = {  # Local training epochs per teacher
        "MNIST": 5 if FAST else 300,
        "CIFAR10": 5 if FAST else 300,
    }

    TEACHER_LR = 0.001  # Learning rate (paper: 0.001)

    TEACHER_BATCH_SIZE = 32  # Batch size (paper: 32)

    # -------------------------------------------------------------------------
    # STUDENT MODEL SETTINGS
    # -------------------------------------------------------------------------
    STUDENT_MODEL = "ResNet18"  # Student architecture (smaller than teacher)

    # -------------------------------------------------------------------------
    # KNOWLEDGE DISTILLATION SETTINGS
    # -------------------------------------------------------------------------
    KD_TEMPERATURE = 4.0  # Softmax temperature (higher = softer targets)

    KD_LEARNING_RATE = 1e-3  # Student learning rate

    KD_EPOCHS = {  # Number of KD epochs
        "MNIST": 6 if FAST else 1000,
        "CIFAR10": 8 if FAST else 30,
    }

    KD_CONFIDENCE_THRESHOLD = 0.35  # Only learn from confident predictions

    KD_GRADIENT_CLIP = 1.0  # Gradient clipping value

    KD_USE_CLASS_AWARE = True  # Use class-aware ensemble (recommended)

    KD_BATCH_SIZE = {  # Batch size for KD
        "MNIST": 64 if FAST else 128,
        "CIFAR10": 64 if FAST else 128,
    }

    # Early stopping settings (for KD training)
    KD_EARLY_STOPPING = True  # Enable early stopping for KD
    KD_PATIENCE = 5  # Stop if no improvement for N epochs
    KD_MIN_DELTA = 0.001  # Minimum improvement to count as progress

    # Early stopping settings (for Teacher training)
    TEACHER_EARLY_STOPPING = True  # Enable early stopping for teachers
    TEACHER_PATIENCE = 10  # Stop if no improvement for N epochs (higher for teachers)
    TEACHER_MIN_DELTA = 0.001  # Minimum improvement to count as progress

    # -------------------------------------------------------------------------
    # SYNTHETIC DATA GENERATION SETTINGS (Data-Free KD)
    # -------------------------------------------------------------------------
    # NOTE: Data-free generation is experimental and may not produce good results.
    # Set to False to use real test data and verify the KD method works.
    USE_SYNTHETIC_DATA = True  # True: Generate synthetic data (data-free)
    # False: Use real test data (RECOMMENDED)

    GENERATOR_LATENT_DIM = 100  # Noise vector dimension

    GENERATOR_EPOCHS = {  # Epochs to train generator
        "MNIST": 50 if FAST else 2000,
        "CIFAR10": 50 if FAST else 1000,
    }

    GENERATOR_BATCH_SIZE = {  # Batch size for generator training
        "MNIST": 8 if FAST else 32,
        "CIFAR10": 8 if FAST else 32,
    }

    GENERATOR_LR = 3e-3  # Generator learning rate

    GENERATOR_DIVERSITY_WEIGHT = 5.0  # Weight for diversity loss

    NUM_KD_BATCHES = {  # Number of synthetic batches for KD
        "MNIST": 100 if FAST else 2000,
        "CIFAR10": 100 if FAST else 1000
    }

    # -------------------------------------------------------------------------
    # VIRTUAL RETRAINING SETTINGS (Optional Phase 3)
    # -------------------------------------------------------------------------
    ENABLE_VIRTUAL_RETRAINING = False  # Usually hurts performance, disabled

    VIRTUAL_EPOCHS = 2 if FAST else 100

    VIRTUAL_LR = 5e-4

    # -------------------------------------------------------------------------
    # END OF CONFIGURATION
    # -------------------------------------------------------------------------

# ============================================================================
# SETUP AND INITIALIZATION
# ============================================================================

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Set random seeds
torch.manual_seed(RANDOM_SEED if 'RANDOM_SEED' in dir() else 42)
np.random.seed(RANDOM_SEED if 'RANDOM_SEED' in dir() else 42)
random.seed(RANDOM_SEED if 'RANDOM_SEED' in dir() else 42)

# Handle configuration for both script and import modes
if 'FAST' not in dir():
    FAST = False
if 'DATASET' not in dir():
    DATASET = "MNIST"
if 'RESULTS_DIR' not in dir():
    RESULTS_DIR = "./results"
if 'ORBIT_LABELS' not in dir():
    ORBIT_LABELS = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 0, 1], [2, 3, 4, 5, 8, 9]]

# Create experiment directory
os.makedirs(RESULTS_DIR, exist_ok=True)

# Include data mode in experiment name for easy identification
_data_mode = "synthetic_data" if USE_SYNTHETIC_DATA else "without_synthetic_data"
EXPERIMENT_NAME = f"KD_{DATASET}_ClassAware_{_data_mode}"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_id = f"{EXPERIMENT_NAME}_{timestamp}"
experiment_dir = os.path.join(RESULTS_DIR, experiment_id)
os.makedirs(experiment_dir, exist_ok=True)

print(f"\n{'=' * 60}")
print(f"EXPERIMENT CONFIGURATION")
print(f"{'=' * 60}")
print(f"Experiment ID: {experiment_id}")
print(f"Dataset: {DATASET}")
print(f"Fast Mode: {FAST}")
print(f"Results Directory: {experiment_dir}")
print(f"{'=' * 60}\n")

# Dataset loading with appropriate transforms
print(f"Loading {DATASET} dataset...")
if DATASET == "MNIST":
    # MNIST: grayscale 28x28 -> resize to 224x224, convert to 3 channels
    tf_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    tf_test = tf_train
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=tf_train)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=tf_test)

elif DATASET == "CIFAR10":
    # CIFAR-10: RGB 32x32 -> resize to 224x224 with augmentation for training
    tf_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])
    tf_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=tf_train)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=tf_test)
else:
    raise ValueError(f"Unknown dataset: {DATASET}. Use 'MNIST' or 'CIFAR10'")

print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

# Class names for reference
# MNIST: 0-9 digits
# CIFAR-10: 0=airplane, 1=automobile, 2=bird, 3=cat, 4=deer,
#           5=dog, 6=frog, 7=horse, 8=ship, 9=truck
CLASS_NAMES = {
    "MNIST": [str(i) for i in range(10)],
    "CIFAR10": ["airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck"]
}

# Cell 2: Non‑IID orbit splits (5 orbits; 2 with 4 classes, 3 with 6 classes)
# Mirrors the paper's non‑IID setting (4 vs 6 classes per orbit).
# Orbit labels are defined in the CONFIGURATION section at the top.

orbit_labels = ORBIT_LABELS  # From main config

# Print orbit class names
print(f"\nOrbit class assignments for {DATASET}:")
for i, labels in enumerate(orbit_labels, 1):
    class_names = [CLASS_NAMES[DATASET][l] for l in labels]
    print(f"  Orbit {i}: {class_names}")


def idxs_for_labels(dataset, labels):
    """Fast label-based indexing using dataset.targets (avoids loading images)."""
    labels_set = set(labels)
    targets = dataset.targets
    if isinstance(targets, torch.Tensor):
        targets = targets.tolist()
    return [i for i, y in enumerate(targets) if y in labels_set]


print("Splitting dataset into orbits...")
train_orbit_subsets, val_orbit_subsets = [], []
for lbls in tqdm(orbit_labels, desc="Building orbit splits"):
    idxs = idxs_for_labels(train_dataset, lbls)
    random.shuffle(idxs)
    split = int(0.85 * len(idxs))  # 85/15 train/val
    tr_idx, va_idx = idxs[:split], idxs[split:]
    if FAST:
        tr_idx = tr_idx[:800]  # shrink per orbit for speed
        va_idx = va_idx[:200]
    train_orbit_subsets.append(Subset(train_dataset, tr_idx))
    val_orbit_subsets.append(Subset(train_dataset, va_idx))

for i, lbls in enumerate(orbit_labels, 1):
    print(f"Orbit {i}: labels {lbls}, train {len(train_orbit_subsets[i - 1])}, val {len(val_orbit_subsets[i - 1])}")


def make_resnet50(n_classes=10):
    m = models.resnet50(weights=None)  # no pretrain (offline-friendly)
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m


def make_resnet18(n_classes=10):
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m


# ============================================================================
# GENERATOR NETWORK FOR DATA-FREE KNOWLEDGE DISTILLATION
# ============================================================================
class Generator(nn.Module):
    """
    Improved Generator network with residual-style blocks and better initialization.
    Generates 224x224 images from random noise for data-free knowledge distillation.
    """

    def __init__(self, latent_dim=100, img_size=224, channels=3, ngf=64):
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
        self.up3 = self._make_up_block(ngf * 2, ngf)  # 56→112
        self.up4 = nn.Sequential(  # 112→224
            nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf // 2, channels, 3, 1, 1, bias=True),
            nn.Tanh()  # Output in [-1, 1]
        )

        # Initialize weights for better training
        self.apply(self._init_weights)

    def _make_up_block(self, in_ch, out_ch):
        """Create upsampling block with extra conv for more capacity."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def _init_weights(self, m):
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

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), self.ngf * 8, self.init_size, self.init_size)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return x

    def generate_batch(self, batch_size, device):
        """Generate a batch of synthetic images."""
        z = torch.randn(batch_size, self.latent_dim, device=device)
        return self(z)


def train_generator(generator, teachers, orbit_labels, epochs=500, batch_size=32,
                    lr=2e-3, diversity_weight=5.0, device='cuda:1'):
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
        epochs: Number of training epochs
        batch_size: Batch size for generation
        lr: Learning rate
        diversity_weight: Weight for diversity loss
        device: Device to train on

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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    print("\n" + "=" * 60)
    print("PHASE 1: Training Generator (Class-Conditional DeepInversion)")
    print("=" * 60)
    print(f"Training for {epochs} epochs with batch_size={batch_size}")
    print("Target: confidence > 0.7, entropy < 0.5")

    n_classes = 10
    avg_confidence = 0.0
    best_confidence = 0.0
    best_state = None

    epoch_pbar = tqdm(range(epochs), desc="Generator Training")
    for epoch in epoch_pbar:
        optimizer.zero_grad()

        # Generate images
        z = torch.randn(batch_size, generator.latent_dim, device=device)
        fake_images = generator(z)

        # =====================================================================
        # CLASS-CONDITIONAL: Target a specific class this epoch for diversity
        # =====================================================================
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

        # =====================================================================
        # LOSS 1: Cross-entropy to target class (class-conditional)
        # =====================================================================
        target_labels = torch.full((batch_size,), target_class, device=device, dtype=torch.long)
        loss_target = F.cross_entropy(torch.log(avg_pred + 1e-8), target_labels)

        # =====================================================================
        # LOSS 2: Entropy minimization (confidence)
        # =====================================================================
        entropy = -(avg_pred * torch.log(avg_pred + 1e-8)).sum(dim=1)
        loss_entropy = entropy.mean()

        # =====================================================================
        # LOSS 3: One-hot encouragement (maximize confidence)
        # =====================================================================
        max_probs = avg_pred.max(dim=1)[0]
        loss_onehot = -max_probs.mean()

        # =====================================================================
        # LOSS 4: Total Variation (smoothness - helps generate cleaner images)
        # =====================================================================
        tv_h = torch.abs(fake_images[:, :, 1:, :] - fake_images[:, :, :-1, :]).mean()
        tv_w = torch.abs(fake_images[:, :, :, 1:] - fake_images[:, :, :, :-1]).mean()
        loss_tv = tv_h + tv_w

        # =====================================================================
        # LOSS 5: L2 regularization (prevent extreme values)
        # =====================================================================
        loss_l2 = (fake_images ** 2).mean()

        # =====================================================================
        # LOSS 6: Feature statistics matching
        # =====================================================================
        img_mean = fake_images.mean()
        img_std = fake_images.std()
        loss_stats = (img_mean ** 2) + ((img_std - 1.0) ** 2)

        # =====================================================================
        # TOTAL LOSS - Weighted combination
        # =====================================================================
        total_loss = (
                1.0 * loss_target +  # Class targeting
                1.0 * loss_entropy +  # Entropy minimization
                3.0 * loss_onehot +  # Confidence maximization (increased weight)
                0.0005 * loss_tv +  # Smoothness
                0.005 * loss_l2 +  # L2 regularization
                0.05 * loss_stats  # Feature matching
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
            tqdm.write(f"Generator Epoch {epoch + 1}: loss={loss_val:.4f}, "
                       f"entropy={entropy_val:.3f}, conf={avg_confidence:.3f}, "
                       f"best={best_confidence:.3f}, lr={scheduler.get_last_lr()[0]:.6f}")

        # Early stopping if confidence is good
        if avg_confidence > 0.80 and entropy_val < 0.4:
            tqdm.write(f"✓ Early stopping: confidence={avg_confidence:.3f}, entropy={entropy_val:.3f}")
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
        print("   For better results, consider using real data (USE_SYNTHETIC_DATA=False)")

    return generator


class SyntheticDataLoader:
    """
    A DataLoader-like object that generates synthetic batches on-the-fly
    using the trained generator.
    """

    def __init__(self, generator, num_batches, batch_size, device):
        self.generator = generator
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.device = device

    def __iter__(self):
        self.generator.eval()
        for _ in range(self.num_batches):
            with torch.no_grad():
                fake_images = self.generator.generate_batch(self.batch_size, self.device)
            # Yield images with dummy labels (not used in KD)
            yield fake_images, torch.zeros(self.batch_size, dtype=torch.long, device=self.device)

    def __len__(self):
        return self.num_batches


def train_teacher(model, train_subset, val_subset, epochs=3, lr=1e-3, bs=64,
                  early_stopping=True, patience=10, min_delta=0.001):
    """Train a teacher model with optional early stopping."""
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = DataLoader(train_subset, batch_size=bs, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=bs, num_workers=2, pin_memory=True)

    # Early stopping setup
    best_val_acc = 0.0
    best_state = None
    patience_counter = 0
    epochs_trained = 0

    epoch_pbar = tqdm(range(epochs), desc="Training", leave=False)
    for ep in epoch_pbar:
        model.train();
        total = 0;
        correct = 0;
        loss_sum = 0.0
        batch_pbar = tqdm(train_loader, desc=f"Epoch {ep + 1}/{epochs}", leave=False)
        for x, y in batch_pbar:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad();
            loss.backward();
            opt.step()
            loss_sum += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
            batch_pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{correct / total:.3f}")
        # val
        model.eval();
        vtot = 0;
        vcor = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                vcor += (out.argmax(1) == y).sum().item()
                vtot += x.size(0)

        val_acc = vcor / vtot
        epochs_trained = ep + 1
        epoch_pbar.set_postfix(train_acc=f"{correct / total:.3f}", val_acc=f"{val_acc:.3f}", best=f"{best_val_acc:.3f}")
        tqdm.write(
            f"Teacher ep {ep + 1}: train loss {loss_sum / total:.3f}, train acc {correct / total:.3f}, val acc {val_acc:.3f}")

        # Early stopping check
        if early_stopping:
            if val_acc > best_val_acc + min_delta:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    tqdm.write(f"  ⚠ Early stopping at epoch {ep + 1}! Best val acc: {best_val_acc:.4f}")
                    if best_state is not None:
                        model.load_state_dict(best_state)
                    break
        else:
            # Just track best
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())

    # Load best model at the end
    if best_state is not None and not early_stopping:
        model.load_state_dict(best_state)

    return model.eval(), epochs_trained, best_val_acc


# Train 5 teachers using settings from main config
_teacher_epochs = TEACHER_EPOCHS.get(DATASET, 300) if isinstance(TEACHER_EPOCHS, dict) else TEACHER_EPOCHS
_teacher_lr = TEACHER_LR if 'TEACHER_LR' in dir() else 0.001
_teacher_bs = TEACHER_BATCH_SIZE if 'TEACHER_BATCH_SIZE' in dir() else 32
_teacher_early_stop = TEACHER_EARLY_STOPPING if 'TEACHER_EARLY_STOPPING' in dir() else True
_teacher_patience = TEACHER_PATIENCE if 'TEACHER_PATIENCE' in dir() else 10
_teacher_min_delta = TEACHER_MIN_DELTA if 'TEACHER_MIN_DELTA' in dir() else 0.001

teachers = []
teacher_training_info = []  # Track epochs trained and best acc for each teacher
print(f"\nTraining teachers (epochs={_teacher_epochs}, lr={_teacher_lr}, bs={_teacher_bs})...")
if _teacher_early_stop:
    print(f"Early stopping enabled: patience={_teacher_patience}, min_delta={_teacher_min_delta}")

teacher_pbar = tqdm(enumerate(zip(train_orbit_subsets, val_orbit_subsets), 1),
                    total=len(train_orbit_subsets), desc="Teachers")
for i, (tr, va) in teacher_pbar:
    teacher_pbar.set_description(f"Teacher {i}/5 (Orbit labels: {orbit_labels[i - 1]})")
    t = make_resnet50()
    t, epochs_trained, best_acc = train_teacher(
        t, tr, va,
        epochs=_teacher_epochs,
        lr=_teacher_lr,
        bs=_teacher_bs,
        early_stopping=_teacher_early_stop,
        patience=_teacher_patience,
        min_delta=_teacher_min_delta
    )
    teachers.append(t)
    teacher_training_info.append({
        "epochs_trained": epochs_trained,
        "best_val_acc": best_acc,
        "early_stopped": epochs_trained < _teacher_epochs
    })
    tqdm.write(f"Teacher {i} complete: {epochs_trained}/{_teacher_epochs} epochs, best val acc: {best_acc:.4f}")

# ============================================================================
# PHASE 1: DATA-FREE SYNTHETIC DATA GENERATION
# ============================================================================
# The ground station has NO real data - only the uploaded teacher models.
# We train a Generator to create synthetic images that teachers agree on.
# This is the key innovation for truly data-free knowledge distillation.

# Get synthetic data settings from main config
_GENERATOR_EPOCHS = GENERATOR_EPOCHS.get(DATASET, 500) if isinstance(GENERATOR_EPOCHS, dict) else GENERATOR_EPOCHS
_GENERATOR_BATCH_SIZE = GENERATOR_BATCH_SIZE.get(DATASET, 32) if isinstance(GENERATOR_BATCH_SIZE,
                                                                            dict) else GENERATOR_BATCH_SIZE
_NUM_KD_BATCHES = NUM_KD_BATCHES.get(DATASET, 500) if isinstance(NUM_KD_BATCHES, dict) else NUM_KD_BATCHES

if USE_SYNTHETIC_DATA:
    print("\n" + "=" * 60)
    print("DATA-FREE MODE: Training generator for synthetic data")
    print("=" * 60)

    # Create and train generator
    _latent_dim = GENERATOR_LATENT_DIM if 'GENERATOR_LATENT_DIM' in dir() else 100
    generator = Generator(latent_dim=_latent_dim, img_size=224, channels=3, ngf=64)

    _gen_lr = GENERATOR_LR if 'GENERATOR_LR' in dir() else 2e-3
    _diversity_weight = GENERATOR_DIVERSITY_WEIGHT if 'GENERATOR_DIVERSITY_WEIGHT' in dir() else 5.0

    generator = train_generator(
        generator=generator,
        teachers=teachers,
        orbit_labels=orbit_labels,
        epochs=_GENERATOR_EPOCHS,
        batch_size=_GENERATOR_BATCH_SIZE,
        lr=_gen_lr,
        diversity_weight=_diversity_weight,
        device=device
    )

    # Create synthetic data loader for KD
    _kd_batch_size = KD_BATCH_SIZE.get(DATASET, 128) if isinstance(KD_BATCH_SIZE, dict) else KD_BATCH_SIZE
    kd_loader = SyntheticDataLoader(
        generator=generator,
        num_batches=_NUM_KD_BATCHES,
        batch_size=_kd_batch_size,
        device=device
    )
    print(f"Synthetic KD loader: {len(kd_loader)} batches of synthetic images")

    # -------------------------------------------------------------------------
    # Save and visualize synthetic images
    # -------------------------------------------------------------------------
    print("\nGenerating and saving synthetic images for visualization...")

    synthetic_dir = os.path.join(experiment_dir, "synthetic_images")
    os.makedirs(synthetic_dir, exist_ok=True)

    # Generate a batch of images for visualization
    generator.eval()
    with torch.no_grad():
        # Generate 64 images for visualization
        z = torch.randn(64, generator.latent_dim, device=device)
        synthetic_images = generator(z)

        # Get teacher predictions for these images
        vote_sum = torch.zeros(64, 10, device=device)
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

    # Denormalize images for visualization
    if DATASET == "MNIST":
        mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
    else:  # CIFAR10
        mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616], device=device).view(1, 3, 1, 1)

    denorm_images = synthetic_images * std + mean
    denorm_images = denorm_images.clamp(0, 1)

    # Save individual synthetic images
    from torchvision.utils import save_image

    for idx in range(min(32, len(denorm_images))):
        img_path = os.path.join(synthetic_dir,
                                f"synthetic_{idx:03d}_class{predicted_classes[idx].item()}_conf{confidences[idx].item():.2f}.png")
        save_image(denorm_images[idx], img_path)

    # Create grid visualization
    fig, axes = plt.subplots(8, 8, figsize=(16, 16))
    fig.suptitle(f'Synthetic Images Generated by Data-Free KD\n{DATASET} Dataset', fontsize=16, fontweight='bold')

    for idx, ax in enumerate(axes.flat):
        if idx < len(denorm_images):
            img = denorm_images[idx].cpu().permute(1, 2, 0).numpy()
            ax.imshow(img)
            pred_class = predicted_classes[idx].item()
            conf = confidences[idx].item()
            class_name = CLASS_NAMES[DATASET][pred_class]
            ax.set_title(f'{class_name}\n({conf * 100:.0f}%)', fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    grid_path = os.path.join(synthetic_dir, "synthetic_grid.png")
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved synthetic image grid: {grid_path}")

    # Save a batch as tensor for later use
    torch.save({
        'images': synthetic_images.cpu(),
        'predicted_classes': predicted_classes.cpu(),
        'confidences': confidences.cpu(),
        'ensemble_probs': ensemble_probs.cpu(),
    }, os.path.join(synthetic_dir, "synthetic_batch.pt"))
    print(f"Saved synthetic batch tensor: {synthetic_dir}/synthetic_batch.pt")

    # Create histogram of predicted classes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Class distribution
    class_counts = torch.bincount(predicted_classes.cpu(), minlength=10).numpy()
    ax1.bar(range(10), class_counts, color='#3498db', edgecolor='black')
    ax1.set_xlabel('Predicted Class', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Distribution of Predicted Classes\nin Synthetic Images', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(10))
    ax1.set_xticklabels(CLASS_NAMES[DATASET], rotation=45, ha='right')

    # Confidence distribution
    conf_values = confidences.cpu().numpy()
    # Use 'auto' bins to handle edge cases with low variance data
    try:
        ax2.hist(conf_values, bins='auto', color='#2ecc71', edgecolor='black', alpha=0.7)
    except ValueError:
        # Fallback: if auto fails, use a simple bar showing the mean
        ax2.bar([0.5], [len(conf_values)], width=0.1, color='#2ecc71', edgecolor='black')
        ax2.set_xlim(0, 1)
    ax2.axvline(x=conf_values.mean(), color='red', linestyle='--', label=f'Mean: {conf_values.mean():.2f}')
    ax2.set_xlabel('Confidence', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Confidence Distribution\nof Teacher Ensemble', fontsize=14, fontweight='bold')
    ax2.legend()

    plt.tight_layout()
    dist_path = os.path.join(synthetic_dir, "synthetic_distribution.png")
    plt.savefig(dist_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved distribution plot: {dist_path}")

    # -------------------------------------------------------------------------
    # Save COMPLETE synthetic dataset used for KD training
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SAVING COMPLETE SYNTHETIC DATASET FOR KD TRAINING")
    print("=" * 60)

    all_synthetic_images = []
    all_pseudo_labels = []
    all_soft_labels = []
    all_confidences = []

    generator.eval()
    with torch.no_grad():
        for batch_idx in tqdm(range(_NUM_KD_BATCHES), desc="Generating & saving synthetic data"):
            z = torch.randn(_kd_batch_size, generator.latent_dim, device=device)
            fake_images = generator(z)

            # Get pseudo-labels from class-aware ensemble
            vote_sum = torch.zeros(fake_images.size(0), 10, device=device)
            vote_count = torch.zeros(10, device=device)
            _temp = KD_TEMPERATURE if 'KD_TEMPERATURE' in dir() else 4.0
            for teacher, labels in zip(teachers, orbit_labels):
                logits = teacher(fake_images)
                probs = F.softmax(logits / _temp, dim=1)  # Use temperature
                for c in labels:
                    vote_sum[:, c] += probs[:, c]
                    vote_count[c] += 1
            vote_count = vote_count.clamp(min=1)
            soft_labels = vote_sum / vote_count.unsqueeze(0)

            pseudo_labels = soft_labels.argmax(dim=1)
            batch_conf, _ = soft_labels.max(dim=1)

            all_synthetic_images.append(fake_images.cpu())
            all_pseudo_labels.append(pseudo_labels.cpu())
            all_soft_labels.append(soft_labels.cpu())
            all_confidences.append(batch_conf.cpu())

            # Clear GPU memory periodically
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()

    # Concatenate all batches
    full_synthetic_dataset = {
        'images': torch.cat(all_synthetic_images, dim=0),
        'pseudo_labels': torch.cat(all_pseudo_labels, dim=0),
        'soft_labels': torch.cat(all_soft_labels, dim=0),
        'confidences': torch.cat(all_confidences, dim=0),
        'num_samples': len(torch.cat(all_pseudo_labels)),
        'temperature': _temp,
        'generator_config': {
            'latent_dim': generator.latent_dim,
            'epochs_trained': _GENERATOR_EPOCHS,
        },
        'class_names': CLASS_NAMES[DATASET],
    }

    # Save complete dataset
    full_dataset_path = os.path.join(synthetic_dir, "full_synthetic_dataset.pt")
    torch.save(full_synthetic_dataset, full_dataset_path)
    print(f"\n✓ Saved COMPLETE synthetic dataset:")
    print(f"  Path: {full_dataset_path}")
    print(f"  Total samples: {full_synthetic_dataset['num_samples']}")
    print(f"  Image shape: {full_synthetic_dataset['images'].shape}")
    print(f"  Soft labels shape: {full_synthetic_dataset['soft_labels'].shape}")

    # Class distribution summary
    class_dist = torch.bincount(full_synthetic_dataset['pseudo_labels'], minlength=10)
    print(f"\n  Class distribution:")
    for i, (name, count) in enumerate(zip(CLASS_NAMES[DATASET], class_dist.tolist())):
        pct = count / full_synthetic_dataset['num_samples'] * 100
        print(f"    {name}: {count} ({pct:.1f}%)")

    print(f"\n  Mean confidence: {full_synthetic_dataset['confidences'].mean():.4f}")
    print(f"  Min confidence: {full_synthetic_dataset['confidences'].min():.4f}")
    print(f"  Max confidence: {full_synthetic_dataset['confidences'].max():.4f}")

    # Save a summary plot of the full dataset
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Class distribution
    axes[0].bar(range(10), class_dist.numpy(), color='#3498db', edgecolor='black')
    axes[0].set_xlabel('Class', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title(f'Full Synthetic Dataset\nClass Distribution (N={full_synthetic_dataset["num_samples"]})',
                      fontsize=14, fontweight='bold')
    axes[0].set_xticks(range(10))
    axes[0].set_xticklabels(CLASS_NAMES[DATASET], rotation=45, ha='right')

    # Plot 2: Confidence distribution
    axes[1].hist(full_synthetic_dataset['confidences'].numpy(), bins=50,
                 color='#2ecc71', edgecolor='black', alpha=0.7)
    axes[1].axvline(x=full_synthetic_dataset['confidences'].mean().item(),
                    color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {full_synthetic_dataset["confidences"].mean():.3f}')
    axes[1].set_xlabel('Confidence', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Confidence Distribution\nof Synthetic Samples', fontsize=14, fontweight='bold')
    axes[1].legend()

    # Plot 3: Sample grid of synthetic images
    sample_indices = torch.randperm(min(64, full_synthetic_dataset['num_samples']))[:64]
    sample_images = full_synthetic_dataset['images'][sample_indices]

    # Denormalize for display
    if DATASET == "MNIST":
        mean = torch.tensor([0.1307]).view(1, 1, 1, 1)
        std = torch.tensor([0.3081]).view(1, 1, 1, 1)
    else:
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)

    sample_images = sample_images * std + mean
    sample_images = sample_images.clamp(0, 1)

    # Create grid
    grid_size = 8
    grid_img = torch.zeros(3, grid_size * 32, grid_size * 32)
    for idx in range(min(64, len(sample_images))):
        row, col = idx // grid_size, idx % grid_size
        img = sample_images[idx]
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        # Resize to 32x32 for display
        img_resized = F.interpolate(img.unsqueeze(0), size=(32, 32), mode='bilinear', align_corners=False)[0]
        grid_img[:, row * 32:(row + 1) * 32, col * 32:(col + 1) * 32] = img_resized

    axes[2].imshow(grid_img.permute(1, 2, 0).numpy())
    axes[2].set_title('Sample Synthetic Images\n(8x8 Grid)', fontsize=14, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()
    full_dist_path = os.path.join(synthetic_dir, "full_dataset_summary.png")
    plt.savefig(full_dist_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved full dataset summary plot: {full_dist_path}")

else:
    print("\nUsing real test data for KD (comparison mode)...")
    generator = None
    kd_loader = DataLoader(test_dataset, batch_size=(64 if FAST else 128),
                           shuffle=True, num_workers=2, pin_memory=True)

# Eval loader always uses real test data (for fair evaluation)
print("Creating evaluation loader with real test data...")
eval_loader = DataLoader(test_dataset, batch_size=(64 if FAST else 128),
                         num_workers=2, pin_memory=True)


# Quick evaluation utilities
def eval_acc(model, loader):
    model.eval();
    tot = 0;
    cor = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            cor += (out.argmax(1) == y).sum().item();
            tot += x.size(0)
    return cor / tot


def eval_ensemble_acc(teachers, loader, weights=None):
    for m in teachers: m.eval()
    tot = 0;
    cor = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = [m(x) for m in teachers]
            if weights is None:
                ens = torch.stack(logits).mean(dim=0)
            else:
                w = torch.tensor(weights, device=device).view(-1, 1, 1)
                ens = (torch.stack(logits) * w).sum(dim=0)
            cor += (ens.argmax(1) == y).sum().item();
            tot += x.size(0)
    return cor / tot


def class_aware_ensemble(teachers, x, orbit_labels):
    """Each teacher contributes only to classes it was trained on."""
    batch_size = x.size(0)
    n_classes = 10

    # Accumulate weighted votes per class
    vote_sum = torch.zeros(batch_size, n_classes, device=device)
    vote_count = torch.zeros(n_classes, device=device)

    for teacher, labels in zip(teachers, orbit_labels):
        logits = teacher(x)
        probs = F.softmax(logits, dim=1)
        for c in labels:
            vote_sum[:, c] += probs[:, c]
            vote_count[c] += 1

    # Average votes per class
    vote_count = vote_count.clamp(min=1)
    ensemble_probs = vote_sum / vote_count.unsqueeze(0)
    return ensemble_probs


def eval_class_aware_ensemble_acc(teachers, loader, orbit_labels):
    """Evaluate class-aware ensemble accuracy."""
    for m in teachers: m.eval()
    tot = 0;
    cor = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            ens_probs = class_aware_ensemble(teachers, x, orbit_labels)
            cor += (ens_probs.argmax(1) == y).sum().item();
            tot += x.size(0)
    return cor / tot


# Teacher and ensemble baseline accuracy (sanity check)
print("Evaluating teachers and ensemble...")
teacher_accs = [eval_acc(t, eval_loader) for t in tqdm(teachers, desc="Evaluating teachers")]
ens_acc = eval_ensemble_acc(teachers, eval_loader)
class_aware_ens_acc = eval_class_aware_ensemble_acc(teachers, eval_loader, orbit_labels)
print("\nTeacher accuracies (on FULL test set - expected low due to non-IID):", teacher_accs)
print("Naive ensemble accuracy:", ens_acc)
print("Class-aware ensemble accuracy:", class_aware_ens_acc)

# Paper's KD objective: R_KL^S = KL(D_teacher, D_student)
# Get KD settings from main config
_kd_epochs_val = KD_EPOCHS.get(DATASET, 20) if isinstance(KD_EPOCHS, dict) else KD_EPOCHS
_kd_temp = KD_TEMPERATURE if 'KD_TEMPERATURE' in dir() else 4.0
_kd_lr = KD_LEARNING_RATE if 'KD_LEARNING_RATE' in dir() else 1e-3
_kd_conf_th = KD_CONFIDENCE_THRESHOLD if 'KD_CONFIDENCE_THRESHOLD' in dir() else 0.35
_kd_clip = KD_GRADIENT_CLIP if 'KD_GRADIENT_CLIP' in dir() else 1.0
_kd_class_aware = KD_USE_CLASS_AWARE if 'KD_USE_CLASS_AWARE' in dir() else True


@dataclass
class KDConfig:
    T: float = _kd_temp  # temperature (higher = softer targets)
    lr: float = _kd_lr  # learning rate
    epochs: int = _kd_epochs_val  # number of epochs
    clip_grad: float = _kd_clip  # gradient clipping
    conf_th: float = _kd_conf_th  # confidence threshold for filtering samples
    use_class_aware: bool = _kd_class_aware  # use class-aware ensemble


student = make_resnet18().to(device)
opt = torch.optim.Adam(student.parameters(), lr=KDConfig.lr)  # Adam works better here

# Early stopping setup
_early_stopping = KD_EARLY_STOPPING if 'KD_EARLY_STOPPING' in dir() else True
_patience = KD_PATIENCE if 'KD_PATIENCE' in dir() else 5
_min_delta = KD_MIN_DELTA if 'KD_MIN_DELTA' in dir() else 0.001

best_acc = 0.0
best_state = None
patience_counter = 0
epochs_trained = 0

print("\nStarting Knowledge Distillation...")
if _early_stopping:
    print(f"Early stopping enabled: patience={_patience}, min_delta={_min_delta}")

kd_epoch_pbar = tqdm(range(KDConfig.epochs), desc="KD Training")
for ep in kd_epoch_pbar:
    student.train();
    loss_sum = 0;
    kept = 0
    kd_batch_pbar = tqdm(kd_loader, desc=f"KD Epoch {ep + 1}/{KDConfig.epochs}", leave=False)
    for x, _ in kd_batch_pbar:
        x = x.to(device)
        with torch.no_grad():
            if KDConfig.use_class_aware:
                # Class-aware ensemble: each teacher contributes only to its classes
                p = class_aware_ensemble(teachers, x, orbit_labels)
            else:
                # Naive average
                logits_list = [m(x) for m in teachers]
                D_teacher = torch.stack(logits_list).mean(dim=0)
                p = F.softmax(D_teacher / KDConfig.T, dim=1)

        D_student = student(x)
        q_log = F.log_softmax(D_student / KDConfig.T, dim=1)

        # Confidence filter: only learn from confident predictions
        maxp, _ = p.max(dim=1)
        mask = (maxp >= KDConfig.conf_th)
        if mask.sum() == 0:
            continue
        kept += mask.sum().item()
        p_sel = p[mask]
        q_log_sel = q_log[mask]

        loss = F.kl_div(q_log_sel, p_sel, reduction='batchmean') * (KDConfig.T ** 2)
        opt.zero_grad();
        loss.backward()
        if KDConfig.clip_grad: nn.utils.clip_grad_norm_(student.parameters(), KDConfig.clip_grad)
        opt.step()
        loss_sum += loss.item()
        kd_batch_pbar.set_postfix(loss=f"{loss.item():.4f}", kept=kept)

    # quick eval
    student.eval();
    tot = 0;
    cor = 0
    with torch.no_grad():
        for x, y in eval_loader:
            x, y = x.to(device), y.to(device)
            out = student(x)
            cor += (out.argmax(1) == y).sum().item();
            tot += x.size(0)

    current_acc = cor / tot
    epochs_trained = ep + 1
    kd_epoch_pbar.set_postfix(acc=f"{current_acc:.3f}", kept=kept, best=f"{best_acc:.3f}")
    tqdm.write(f"KD Epoch {ep + 1}: train KL {loss_sum / max(1, kept):.4f}, eval acc {current_acc:.3f}, kept {kept}")

    # Early stopping check
    if _early_stopping:
        if current_acc > best_acc + _min_delta:
            best_acc = current_acc
            best_state = copy.deepcopy(student.state_dict())
            patience_counter = 0
            tqdm.write(f"  ✓ New best accuracy: {best_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= _patience:
                tqdm.write(f"\n⚠ Early stopping triggered! No improvement for {_patience} epochs.")
                tqdm.write(f"  Best accuracy: {best_acc:.4f}")
                # Restore best model
                if best_state is not None:
                    student.load_state_dict(best_state)
                break
    else:
        # Just track best without early stopping
        if current_acc > best_acc:
            best_acc = current_acc
            best_state = copy.deepcopy(student.state_dict())

# Ensure we use the best model
if best_state is not None and not _early_stopping:
    student.load_state_dict(best_state)
    print(f"Loaded best model (acc: {best_acc:.4f})")

final_acc = eval_acc(student, eval_loader)
print(f"Final student accuracy: {final_acc:.4f} (best during training: {best_acc:.4f})")


# OPTIONAL: Server-local virtual retraining (Phase 3 idea from the paper). [1](https://o365khu-my.sharepoint.com/personal/2025315503_office_khu_ac_kr/Documents/Microsoft%20Copilot%20Chat%20Files/One-Shot%20Federated%20Learning%20for%20LEO%20Constellations.pdf)
# Clone several virtual students, train each on an "orbit-style" partition of MNIST test, then average.

def subset_by_labels(dataset, labels, limit=None):
    """Fast label-based subsetting using dataset.targets."""
    labels_set = set(labels)
    targets = dataset.targets
    if isinstance(targets, torch.Tensor):
        targets = targets.tolist()
    idxs = [i for i, y in enumerate(targets) if y in labels_set]
    if FAST and limit: idxs = idxs[:limit]
    return Subset(dataset, idxs)


parts = [
    subset_by_labels(test_dataset, orbit_labels[2], limit=1000),  # mimic a 6-class partition
    subset_by_labels(test_dataset, orbit_labels[3], limit=1000),
    subset_by_labels(test_dataset, orbit_labels[4], limit=1000),
]


def train_virtual(init_model, ds, epochs=(1 if FAST else 3), lr=5e-4, bs=64):
    m = copy.deepcopy(init_model).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=2, pin_memory=True)
    for ep in range(epochs):
        m.train()
        for x, y in tqdm(loader, desc=f"Virtual ep {ep + 1}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(m(x), y)
            opt.zero_grad();
            loss.backward();
            opt.step()
    return m.eval()


def avg_state_dicts(dicts):
    avg = {}
    for k in dicts[0].keys():
        avg[k] = sum(d[k] for d in dicts) / len(dicts)
    return avg


# Virtual retraining (disabled by default - often hurts because training on
# non-IID partitions causes the student to forget classes)
# Setting is in main config: ENABLE_VIRTUAL_RETRAINING

_virtual_epochs = VIRTUAL_EPOCHS if 'VIRTUAL_EPOCHS' in dir() else (2 if FAST else 3)
_virtual_lr = VIRTUAL_LR if 'VIRTUAL_LR' in dir() else 5e-4

final_student_acc = eval_acc(student, eval_loader)

if ENABLE_VIRTUAL_RETRAINING:
    print("\nStarting virtual retraining on student...")
    virtual_states = []
    for ds in tqdm(parts, desc="Virtual Retraining"):
        vm = train_virtual(student, ds, epochs=_virtual_epochs, lr=_virtual_lr, bs=64)
        virtual_states.append(vm.state_dict())
    student.load_state_dict(avg_state_dicts(virtual_states))
    virtual_retrain_acc = eval_acc(student, eval_loader)
    print("After virtual retraining, student accuracy:", virtual_retrain_acc)
else:
    virtual_retrain_acc = None
    print("\nVirtual retraining disabled (set ENABLE_VIRTUAL_RETRAINING=True to enable)")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "=" * 60)
print("SAVING RESULTS")
print("=" * 60)

# Compile all results
results = {
    "experiment_id": experiment_id,
    "timestamp": timestamp,
    "config": {
        "dataset": DATASET,
        "fast_mode": FAST,
        "device": str(device),
        "num_orbits": len(orbit_labels),
        "orbit_labels": orbit_labels,
        "teacher_config": {
            "model": "ResNet50",
            "epochs": TEACHER_EPOCHS.get(DATASET, 300),
            "learning_rate": TEACHER_LR,
            "batch_size": TEACHER_BATCH_SIZE,
            "early_stopping": _teacher_early_stop,
            "early_stopping_patience": _teacher_patience if _teacher_early_stop else None,
            "early_stopping_min_delta": _teacher_min_delta if _teacher_early_stop else None,
            "training_info": teacher_training_info,  # Epochs and acc for each teacher
        },
        "student_model": "ResNet18",
        "kd_config": {
            "temperature": KDConfig.T,
            "learning_rate": KDConfig.lr,
            "epochs": KDConfig.epochs,
            "clip_grad": KDConfig.clip_grad,
            "confidence_threshold": KDConfig.conf_th,
            "use_class_aware_ensemble": KDConfig.use_class_aware,
            "early_stopping": _early_stopping,
            "early_stopping_patience": _patience if _early_stopping else None,
            "early_stopping_min_delta": _min_delta if _early_stopping else None,
        },
        "synthetic_data": {
            "use_synthetic_data": USE_SYNTHETIC_DATA,
            "generator_epochs": GENERATOR_EPOCHS if USE_SYNTHETIC_DATA else None,
            "generator_batch_size": GENERATOR_BATCH_SIZE if USE_SYNTHETIC_DATA else None,
            "num_kd_batches": NUM_KD_BATCHES if USE_SYNTHETIC_DATA else None,
        },
        "virtual_retraining_enabled": ENABLE_VIRTUAL_RETRAINING,
    },
    "metrics": {
        "teacher_accuracies": teacher_accs,
        "naive_ensemble_accuracy": ens_acc,
        "class_aware_ensemble_accuracy": class_aware_ens_acc,
        "final_student_accuracy": final_student_acc,
        "best_student_accuracy": best_acc,
        "epochs_trained": epochs_trained,
        "early_stopped": epochs_trained < KDConfig.epochs,
        "virtual_retrain_accuracy": virtual_retrain_acc,
    },
    "orbit_splits": {
        f"orbit_{i + 1}": {
            "labels": orbit_labels[i],
            "train_size": len(train_orbit_subsets[i]),
            "val_size": len(val_orbit_subsets[i]),
        }
        for i in range(len(orbit_labels))
    },
}

# Save results as JSON
results_path = os.path.join(experiment_dir, "results.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved to: {results_path}")

# Save student model
student_model_path = os.path.join(experiment_dir, "student_model.pt")
torch.save({
    "model_state_dict": student.state_dict(),
    "accuracy": final_student_acc,
}, student_model_path)
print(f"Student model saved to: {student_model_path}")

# Save generator model (if synthetic data was used)
if USE_SYNTHETIC_DATA and generator is not None:
    generator_model_path = os.path.join(experiment_dir, "generator_model.pt")
    torch.save({
        "model_state_dict": generator.state_dict(),
        "latent_dim": generator.latent_dim,
        "img_size": generator.img_size,
    }, generator_model_path)
    print(f"Generator model saved to: {generator_model_path}")

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

# Print summary
print("\n" + "=" * 60)
print("EXPERIMENT SUMMARY")
print("=" * 60)
print(f"Experiment ID: {experiment_id}")
print(f"Dataset: {DATASET}")
print(f"Data-Free Mode: {'YES (synthetic data)' if USE_SYNTHETIC_DATA else 'NO (real test data)'}")
if USE_SYNTHETIC_DATA:
    print(f"  Generator epochs: {GENERATOR_EPOCHS}")
    print(f"  KD batches: {NUM_KD_BATCHES}")
print(f"Teacher accuracies: {[f'{a:.4f}' for a in teacher_accs]}")
# Show teacher early stopping info
for i, info in enumerate(teacher_training_info):
    status = " (early stopped)" if info["early_stopped"] else ""
    print(
        f"  Teacher {i + 1}: {info['epochs_trained']}/{_teacher_epochs} epochs, best val acc: {info['best_val_acc']:.4f}{status}")
print(f"Naive ensemble accuracy: {ens_acc:.4f}")
print(f"Class-aware ensemble accuracy: {class_aware_ens_acc:.4f}")
print(f"Final student accuracy: {final_student_acc:.4f}")
print(f"Best student accuracy: {best_acc:.4f}")
print(f"Epochs trained: {epochs_trained}/{KDConfig.epochs}" + (
    " (early stopped)" if epochs_trained < KDConfig.epochs else ""))
if virtual_retrain_acc:
    print(f"After virtual retraining: {virtual_retrain_acc:.4f}")
print(f"\nAll results saved to: {experiment_dir}")
print("=" * 60)

# ============================================================================
# PLOTTING RESULTS
# ============================================================================
print("\n" + "=" * 60)
print("GENERATING PLOTS")
print("=" * 60)

plots_dir = os.path.join(experiment_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# -----------------------------------------------------------------------------
# Plot 1: Model Accuracy Comparison
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6))

# Prepare data
models = [f'Teacher {i + 1}\n(Orbit {i + 1})' for i in range(len(teacher_accs))]
models += ['Naive\nEnsemble', 'Class-Aware\nEnsemble', 'Student']
accuracies = teacher_accs + [ens_acc, class_aware_ens_acc, final_student_acc]

# Colors
colors = ['#3498db'] * len(teacher_accs) + ['#e74c3c', '#2ecc71', '#9b59b6']

bars = ax.bar(models, [a * 100 for a in accuracies], color=colors, edgecolor='black', linewidth=1.2)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax.annotate(f'{acc * 100:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_ylabel('Accuracy (%)', fontsize=14)
ax.set_xlabel('Model', fontsize=14)
ax.set_title(f'{DATASET} - Model Accuracy Comparison\n({"Data-Free" if USE_SYNTHETIC_DATA else "With Real Data"})',
             fontsize=16, fontweight='bold')
ax.set_ylim(0, 105)
ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)

# Add legend
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor='#3498db', label='Individual Teachers'),
    Patch(facecolor='#e74c3c', label='Naive Ensemble'),
    Patch(facecolor='#2ecc71', label='Class-Aware Ensemble'),
    Patch(facecolor='#9b59b6', label='Distilled Student'),
]
ax.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plot_path = os.path.join(plots_dir, "accuracy_comparison.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path}")

# -----------------------------------------------------------------------------
# Plot 2: Orbit Class Distribution
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6))

class_names = CLASS_NAMES[DATASET]
orbit_matrix = np.zeros((len(orbit_labels), 10))

for i, labels in enumerate(orbit_labels):
    for l in labels:
        orbit_matrix[i, l] = 1

im = ax.imshow(orbit_matrix, cmap='Blues', aspect='auto')

# Labels
ax.set_xticks(range(10))
ax.set_xticklabels(class_names, rotation=45, ha='right')
ax.set_yticks(range(len(orbit_labels)))
ax.set_yticklabels([f'Orbit {i + 1}\n({len(orbit_labels[i])} classes)' for i in range(len(orbit_labels))])

ax.set_xlabel('Classes', fontsize=14)
ax.set_ylabel('Orbits', fontsize=14)
ax.set_title(f'{DATASET} - Non-IID Data Distribution Across Orbits', fontsize=16, fontweight='bold')

# Add text annotations
for i in range(len(orbit_labels)):
    for j in range(10):
        text = '✓' if orbit_matrix[i, j] == 1 else ''
        ax.text(j, i, text, ha='center', va='center', fontsize=14)

plt.colorbar(im, ax=ax, label='Has Class', shrink=0.8)
plt.tight_layout()
plot_path = os.path.join(plots_dir, "orbit_distribution.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path}")

# -----------------------------------------------------------------------------
# Plot 3: Teacher Accuracies with Orbit Info
# -----------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left plot: Teacher accuracies
x = np.arange(len(teacher_accs))
bars = ax1.bar(x, [a * 100 for a in teacher_accs], color='#3498db', edgecolor='black')

# Add accuracy labels
for bar, acc in zip(bars, teacher_accs):
    height = bar.get_height()
    ax1.annotate(f'{acc * 100:.1f}%',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3),
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

ax1.set_xlabel('Teacher (Orbit)', fontsize=12)
ax1.set_ylabel('Accuracy on Full Test Set (%)', fontsize=12)
ax1.set_title('Individual Teacher Accuracies\n(Evaluated on ALL 10 classes)', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels([f'T{i + 1}\n{len(orbit_labels[i])} cls' for i in range(len(teacher_accs))])
ax1.set_ylim(0, 100)

# Right plot: Classes per orbit
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
plot_path = os.path.join(plots_dir, "teacher_analysis.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path}")

# -----------------------------------------------------------------------------
# Plot 4: Ensemble vs Student Comparison
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

methods = ['Best Single\nTeacher', 'Naive\nEnsemble', 'Class-Aware\nEnsemble', 'Distilled\nStudent']
accs = [max(teacher_accs), ens_acc, class_aware_ens_acc, final_student_acc]
colors = ['#95a5a6', '#e74c3c', '#2ecc71', '#9b59b6']

bars = ax.barh(methods, [a * 100 for a in accs], color=colors, edgecolor='black', height=0.6)

# Add value labels
for bar, acc in zip(bars, accs):
    width = bar.get_width()
    ax.annotate(f'{acc * 100:.1f}%',
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(5, 0),
                textcoords="offset points",
                ha='left', va='center', fontsize=12, fontweight='bold')

ax.set_xlabel('Accuracy (%)', fontsize=14)
ax.set_title(
    f'{DATASET} - Knowledge Aggregation Methods\n({"Data-Free Mode" if USE_SYNTHETIC_DATA else "With Real Data"})',
    fontsize=16, fontweight='bold')
ax.set_xlim(0, 110)
ax.axvline(x=100, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plot_path = os.path.join(plots_dir, "aggregation_comparison.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path}")

# -----------------------------------------------------------------------------
# Plot 5: Real vs Synthetic Images Comparison (if synthetic data was used)
# -----------------------------------------------------------------------------
if USE_SYNTHETIC_DATA and generator is not None:
    fig, axes = plt.subplots(2, 8, figsize=(20, 6))
    fig.suptitle(f'{DATASET} - Real vs Synthetic Images Comparison', fontsize=16, fontweight='bold')

    # Get 8 real images
    real_iter = iter(eval_loader)
    real_batch, real_labels = next(real_iter)

    # Denormalize
    if DATASET == "MNIST":
        mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    else:
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)

    real_denorm = real_batch[:8] * std + mean
    real_denorm = real_denorm.clamp(0, 1)

    # Generate 8 synthetic images
    generator.eval()
    with torch.no_grad():
        z = torch.randn(8, generator.latent_dim, device=device)
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

    # Plot real images (top row)
    for i in range(8):
        ax = axes[0, i]
        img = real_denorm[i].permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_title(f'Real: {CLASS_NAMES[DATASET][real_labels[i].item()]}', fontsize=10)
        ax.axis('off')
        if i == 0:
            ax.set_ylabel('REAL', fontsize=12, fontweight='bold')

    # Plot synthetic images (bottom row)
    for i in range(8):
        ax = axes[1, i]
        img = synth_denorm[i].permute(1, 2, 0).numpy()
        ax.imshow(img)
        pred_class = CLASS_NAMES[DATASET][synth_preds[i].item()]
        conf = synth_confs[i].item()
        ax.set_title(f'Synth: {pred_class}\n({conf * 100:.0f}%)', fontsize=10)
        ax.axis('off')
        if i == 0:
            ax.set_ylabel('SYNTHETIC', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "real_vs_synthetic.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")

# -----------------------------------------------------------------------------
# Plot 6: Summary Dashboard
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(16, 10))

# Title
fig.suptitle(f'{DATASET} One-Shot Federated Learning Results\n{experiment_id}',
             fontsize=18, fontweight='bold', y=0.98)

# Subplot 1: Accuracy bars
ax1 = fig.add_subplot(2, 2, 1)
models_short = [f'T{i + 1}' for i in range(len(teacher_accs))] + ['Naive', 'ClassAware', 'Student']
all_accs = teacher_accs + [ens_acc, class_aware_ens_acc, final_student_acc]
colors = ['#3498db'] * len(teacher_accs) + ['#e74c3c', '#2ecc71', '#9b59b6']
ax1.bar(models_short, [a * 100 for a in all_accs], color=colors, edgecolor='black')
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('Model Accuracies')
ax1.set_ylim(0, 105)
for i, acc in enumerate(all_accs):
    ax1.text(i, acc * 100 + 2, f'{acc * 100:.1f}', ha='center', fontsize=9)

# Subplot 2: Orbit heatmap
ax2 = fig.add_subplot(2, 2, 2)
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
Dataset: {DATASET}
Data-Free Mode: {'Yes' if USE_SYNTHETIC_DATA else 'No'}
Number of Orbits: {len(orbit_labels)}
Teacher Model: ResNet-50
Student Model: ResNet-18

KEY RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━
Best Teacher: {max(teacher_accs) * 100:.1f}%
Naive Ensemble: {ens_acc * 100:.1f}%
Class-Aware Ensemble: {class_aware_ens_acc * 100:.1f}%
Student (Final): {final_student_acc * 100:.1f}%

IMPROVEMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━
Ensemble vs Best Teacher: +{(class_aware_ens_acc - max(teacher_accs)) * 100:.1f}%
Student vs Best Teacher: {'+' if final_student_acc > max(teacher_accs) else ''}{(final_student_acc - max(teacher_accs)) * 100:.1f}%
"""
ax3.text(0.1, 0.95, metrics_text, transform=ax3.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Subplot 4: Comparison chart
ax4 = fig.add_subplot(2, 2, 4)
comparison = ['Best Teacher', 'Class-Aware Ens.', 'Student']
comp_accs = [max(teacher_accs) * 100, class_aware_ens_acc * 100, final_student_acc * 100]
comp_colors = ['#3498db', '#2ecc71', '#9b59b6']
wedges, texts, autotexts = ax4.pie(comp_accs, labels=comparison, colors=comp_colors,
                                   autopct='%1.1f%%', startangle=90,
                                   explode=(0, 0.05, 0.1))
ax4.set_title('Accuracy Distribution')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plot_path = os.path.join(plots_dir, "summary_dashboard.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path}")

print(f"\nAll plots saved to: {plots_dir}/")
print("=" * 60)
