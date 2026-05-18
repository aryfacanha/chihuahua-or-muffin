from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

BATCH_SIZE = 32
NUM_WORKERS = 0


def create_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def create_datasets(transform):
    train_dataset = datasets.ImageFolder(
        root=PROCESSED_DATA_DIR / "train",
        transform=transform,
    )
    val_dataset = datasets.ImageFolder(
        root=PROCESSED_DATA_DIR / "val",
        transform=transform,
    )
    test_dataset = datasets.ImageFolder(
        root=PROCESSED_DATA_DIR / "test",
        transform=transform,
    )

    return train_dataset, val_dataset, test_dataset


def create_dataloaders(train_dataset, val_dataset, test_dataset):
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    return train_loader, val_loader, test_loader


def print_dataset_info(train_dataset, val_dataset, test_dataset, train_loader):
    images, labels = next(iter(train_loader))

    print("Classes detectadas:")
    print(train_dataset.classes)

    print("\nMapeamento class_to_idx:")
    print(train_dataset.class_to_idx)

    print("\nQuantidade de imagens por split:")
    print(f"train: {len(train_dataset)}")
    print(f"val: {len(val_dataset)}")
    print(f"test: {len(test_dataset)}")

    print("\nShape de um batch de imagens:")
    print(images.shape)

    print("\nShape das labels:")
    print(labels.shape)


def main():
    transform = create_transform()
    train_dataset, val_dataset, test_dataset = create_datasets(transform)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
    )

    print_dataset_info(train_dataset, val_dataset, test_dataset, train_loader)


if __name__ == "__main__":
    main()
