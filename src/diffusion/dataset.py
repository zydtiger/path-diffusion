from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import PathMNIST


def get_dataset(batch_size: int = 64) -> tuple[DataLoader, DataLoader]:
    """Load PathMNIST dataset and return train/test dataloaders."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 2 * x - 1),
        ]
    )

    train_dataset = PathMNIST(split="train", download=True)
    test_dataset = PathMNIST(split="test", download=True)

    train_dataset.transform = transform
    test_dataset.transform = transform

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, test_loader
