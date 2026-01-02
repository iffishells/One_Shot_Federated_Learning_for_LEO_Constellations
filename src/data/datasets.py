"""
Dataset loading and transformation utilities.
"""

from typing import Tuple
from torchvision import datasets, transforms


# Class names for reference
CLASS_NAMES = {
    "MNIST": [str(i) for i in range(10)],
    "CIFAR10": ["airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck"]
}


def get_transforms(dataset: str, train: bool = True) -> transforms.Compose:
    """
    Get appropriate transforms for the specified dataset.
    
    Args:
        dataset: Dataset name ('MNIST' or 'CIFAR10')
        train: Whether to use training transforms (with augmentation)
    
    Returns:
        Composed transforms
    """
    if dataset == "MNIST":
        # MNIST: grayscale 28x28 -> resize to 224x224, convert to 3 channels
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        return transform
    
    elif dataset == "CIFAR10":
        if train:
            # CIFAR-10 training: with augmentation
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                                   std=[0.2470, 0.2435, 0.2616])
            ])
        else:
            # CIFAR-10 test: no augmentation
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                                   std=[0.2470, 0.2435, 0.2616])
            ])
        return transform
    
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use 'MNIST' or 'CIFAR10'")


def load_dataset(dataset: str, data_root: str = "./data") -> Tuple:
    """
    Load the specified dataset with appropriate transforms.
    
    Args:
        dataset: Dataset name ('MNIST' or 'CIFAR10')
        data_root: Root directory for datasets
    
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    tf_train = get_transforms(dataset, train=True)
    tf_test = get_transforms(dataset, train=False)
    
    if dataset == "MNIST":
        train_dataset = datasets.MNIST(
            root=data_root, train=True, download=True, transform=tf_train
        )
        test_dataset = datasets.MNIST(
            root=data_root, train=False, download=True, transform=tf_test
        )
    elif dataset == "CIFAR10":
        train_dataset = datasets.CIFAR10(
            root=data_root, train=True, download=True, transform=tf_train
        )
        test_dataset = datasets.CIFAR10(
            root=data_root, train=False, download=True, transform=tf_test
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use 'MNIST' or 'CIFAR10'")
    
    print(f"Loaded {dataset} - Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    
    return train_dataset, test_dataset


def get_normalization_stats(dataset: str) -> Tuple:
    """
    Get normalization mean and std for the dataset.
    
    Args:
        dataset: Dataset name
    
    Returns:
        Tuple of (mean, std) as lists
    """
    if dataset == "MNIST":
        return [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    elif dataset == "CIFAR10":
        return [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

