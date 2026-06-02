from pathlib import Path

from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets, transforms

from config import (
    BATCH_SIZE,
    CLASSES,
    HARD_CASES_DATA_DIR,
    IMAGE_SIZE,
    IMAGE_EXTENSIONS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_WORKERS,
    PROCESSED_DATA_DIR,
)


def create_transform(augment=False):
    if augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(
                IMAGE_SIZE,
                scale=(0.85, 1.0),
                ratio=(0.9, 1.1),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ])

    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])


def has_supported_images(directory):
    directory = Path(directory)

    if not directory.exists():
        return False

    return any(
        path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        for path in directory.rglob("*")
    )


def validate_class_mapping(dataset, expected_mapping):
    if dataset.class_to_idx != expected_mapping:
        raise ValueError(
            "Mapeamento de classes inconsistente. "
            f"Esperado: {expected_mapping}. Encontrado: {dataset.class_to_idx}."
        )


def create_optional_hard_cases_dataset(split, transform, expected_mapping):
    split_dir = HARD_CASES_DATA_DIR / split

    if not split_dir.exists():
        return None

    missing_classes = [
        class_name
        for class_name in CLASSES
        if not (split_dir / class_name).is_dir()
    ]
    if missing_classes:
        missing = ", ".join(missing_classes)
        raise FileNotFoundError(
            f"Hard cases de {split} incompletos. Pastas ausentes: {missing}"
        )

    if not has_supported_images(split_dir):
        print(f"Nenhuma imagem de hard cases encontrada em: {split_dir}")
        return None

    hard_cases_dataset = datasets.ImageFolder(
        root=split_dir,
        transform=transform,
    )
    validate_class_mapping(hard_cases_dataset, expected_mapping)

    return hard_cases_dataset


def maybe_combine_with_hard_cases(base_dataset, split, transform, include_hard_cases):
    if not include_hard_cases:
        return base_dataset

    hard_cases_dataset = create_optional_hard_cases_dataset(
        split,
        transform,
        base_dataset.class_to_idx,
    )

    if hard_cases_dataset is None:
        return base_dataset

    print(
        f"Hard cases adicionados ao split {split}: "
        f"{len(hard_cases_dataset)} imagens"
    )
    return ConcatDataset([base_dataset, hard_cases_dataset])


def create_datasets(
    transform=None,
    train_transform=None,
    eval_transform=None,
    include_hard_cases=False,
):
    if transform is not None:
        train_transform = transform
        eval_transform = transform

    train_transform = train_transform or create_transform(augment=True)
    eval_transform = eval_transform or create_transform(augment=False)

    train_dataset = datasets.ImageFolder(
        root=PROCESSED_DATA_DIR / "train",
        transform=train_transform,
    )
    val_dataset = datasets.ImageFolder(
        root=PROCESSED_DATA_DIR / "val",
        transform=eval_transform,
    )
    test_dataset = datasets.ImageFolder(
        root=PROCESSED_DATA_DIR / "test",
        transform=eval_transform,
    )

    val_dataset = maybe_combine_with_hard_cases(
        val_dataset,
        "val",
        eval_transform,
        include_hard_cases,
    )
    train_dataset = maybe_combine_with_hard_cases(
        train_dataset,
        "train",
        train_transform,
        include_hard_cases,
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
    base_train_dataset = (
        train_dataset.datasets[0]
        if isinstance(train_dataset, ConcatDataset)
        else train_dataset
    )

    print("Classes detectadas:")
    print(base_train_dataset.classes)

    print("\nMapeamento class_to_idx:")
    print(base_train_dataset.class_to_idx)

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
