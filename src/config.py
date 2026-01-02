"""
Configuration module for Knowledge Distillation experiments.

All hyperparameters and settings are defined here for easy modification.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional


@dataclass
class TeacherConfig:
    """Configuration for teacher model training."""
    model: str = "ResNet50"
    epochs: Dict[str, int] = field(default_factory=lambda: {"MNIST": 300, "CIFAR10": 300})
    learning_rate: float = 0.001
    batch_size: int = 32
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 0.001


@dataclass
class GeneratorConfig:
    """Configuration for synthetic data generator."""
    latent_dim: int = 100
    epochs: Dict[str, int] = field(default_factory=lambda: {"MNIST": 2000, "CIFAR10": 1000})
    batch_size: Dict[str, int] = field(default_factory=lambda: {"MNIST": 32, "CIFAR10": 32})
    learning_rate: float = 3e-3
    diversity_weight: float = 5.0
    num_kd_batches: Dict[str, int] = field(default_factory=lambda: {"MNIST": 2000, "CIFAR10": 1000})


@dataclass
class KDConfig:
    """Configuration for knowledge distillation."""
    temperature: float = 4.0
    learning_rate: float = 1e-3
    epochs: Dict[str, int] = field(default_factory=lambda: {"MNIST": 1000, "CIFAR10": 30})
    batch_size: Dict[str, int] = field(default_factory=lambda: {"MNIST": 128, "CIFAR10": 128})
    confidence_threshold: float = 0.35
    gradient_clip: float = 1.0
    use_class_aware: bool = True
    early_stopping: bool = True
    patience: int = 5
    min_delta: float = 0.001


@dataclass
class VirtualRetrainingConfig:
    """Configuration for optional virtual retraining phase."""
    enabled: bool = False
    epochs: int = 100
    learning_rate: float = 5e-4


@dataclass
class Config:
    """
    Main configuration class for the entire experiment.
    
    Example usage:
        config = Config()
        config = Config(dataset="CIFAR10", fast_mode=True)
        config = Config.from_fast_mode()
    """
    # General settings
    dataset: str = "MNIST"
    fast_mode: bool = False
    random_seed: int = 42
    results_dir: str = "./results"
    device: str = "cuda:1"
    
    # Orbit configuration (non-IID data distribution)
    orbit_labels: List[List[int]] = field(default_factory=lambda: [
        [0, 1, 2, 3],        # Orbit 1: 4 classes
        [4, 5, 6, 7],        # Orbit 2: 4 classes
        [0, 1, 2, 3, 4, 5],  # Orbit 3: 6 classes
        [6, 7, 8, 9, 0, 1],  # Orbit 4: 6 classes
        [2, 3, 4, 5, 8, 9],  # Orbit 5: 6 classes
    ])
    train_val_split: float = 0.85
    
    # Sub-configurations
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    kd: KDConfig = field(default_factory=KDConfig)
    virtual_retraining: VirtualRetrainingConfig = field(default_factory=VirtualRetrainingConfig)
    
    # Data mode
    use_synthetic_data: bool = True
    student_model: str = "ResNet18"
    
    def __post_init__(self):
        """Apply fast mode adjustments if enabled."""
        if self.fast_mode:
            self._apply_fast_mode()
    
    def _apply_fast_mode(self):
        """Reduce epochs and batch sizes for quick testing."""
        # Teacher settings
        self.teacher.epochs = {"MNIST": 5, "CIFAR10": 5}
        
        # Generator settings
        self.generator.epochs = {"MNIST": 50, "CIFAR10": 50}
        self.generator.batch_size = {"MNIST": 8, "CIFAR10": 8}
        self.generator.num_kd_batches = {"MNIST": 100, "CIFAR10": 100}
        
        # KD settings
        self.kd.epochs = {"MNIST": 6, "CIFAR10": 8}
        self.kd.batch_size = {"MNIST": 64, "CIFAR10": 64}
        
        # Virtual retraining
        self.virtual_retraining.epochs = 2
    
    @classmethod
    def from_fast_mode(cls, dataset: str = "MNIST") -> "Config":
        """Create a fast-mode configuration for quick testing."""
        return cls(dataset=dataset, fast_mode=True)
    
    def get_teacher_epochs(self) -> int:
        """Get teacher epochs for the current dataset."""
        return self.teacher.epochs.get(self.dataset, 300)
    
    def get_generator_epochs(self) -> int:
        """Get generator epochs for the current dataset."""
        return self.generator.epochs.get(self.dataset, 500)
    
    def get_generator_batch_size(self) -> int:
        """Get generator batch size for the current dataset."""
        return self.generator.batch_size.get(self.dataset, 32)
    
    def get_kd_epochs(self) -> int:
        """Get KD epochs for the current dataset."""
        return self.kd.epochs.get(self.dataset, 20)
    
    def get_kd_batch_size(self) -> int:
        """Get KD batch size for the current dataset."""
        return self.kd.batch_size.get(self.dataset, 128)
    
    def get_num_kd_batches(self) -> int:
        """Get number of KD batches for the current dataset."""
        return self.generator.num_kd_batches.get(self.dataset, 500)
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary for saving."""
        return {
            "dataset": self.dataset,
            "fast_mode": self.fast_mode,
            "random_seed": self.random_seed,
            "results_dir": self.results_dir,
            "device": self.device,
            "num_orbits": len(self.orbit_labels),
            "orbit_labels": self.orbit_labels,
            "train_val_split": self.train_val_split,
            "use_synthetic_data": self.use_synthetic_data,
            "student_model": self.student_model,
            "teacher_config": asdict(self.teacher),
            "generator_config": asdict(self.generator),
            "kd_config": asdict(self.kd),
            "virtual_retraining_config": asdict(self.virtual_retraining),
        }
    
    def print_config(self):
        """Print the configuration in a readable format."""
        print(f"\n{'=' * 60}")
        print("EXPERIMENT CONFIGURATION")
        print(f"{'=' * 60}")
        print(f"Dataset: {self.dataset}")
        print(f"Fast Mode: {self.fast_mode}")
        print(f"Device: {self.device}")
        print(f"Results Directory: {self.results_dir}")
        print(f"Use Synthetic Data: {self.use_synthetic_data}")
        print(f"\nTeacher Config:")
        print(f"  Model: {self.teacher.model}")
        print(f"  Epochs: {self.get_teacher_epochs()}")
        print(f"  Learning Rate: {self.teacher.learning_rate}")
        print(f"  Batch Size: {self.teacher.batch_size}")
        print(f"  Early Stopping: {self.teacher.early_stopping}")
        print(f"\nKD Config:")
        print(f"  Temperature: {self.kd.temperature}")
        print(f"  Epochs: {self.get_kd_epochs()}")
        print(f"  Learning Rate: {self.kd.learning_rate}")
        print(f"  Class-Aware Ensemble: {self.kd.use_class_aware}")
        print(f"{'=' * 60}\n")

