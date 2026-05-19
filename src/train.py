from pathlib import Path

import torch
from torch import nn, optim

from dataset import create_dataloaders, create_datasets, create_transform
from model import create_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"

EPOCHS = 5
LEARNING_RATE = 0.001


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    correct_predictions = 0
    total_images = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total_images += labels.size(0)

    average_loss = total_loss / total_images
    accuracy = correct_predictions / total_images

    return average_loss, accuracy


def validate(model, val_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_images = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_images += labels.size(0)

    average_loss = total_loss / total_images
    accuracy = correct_predictions / total_images

    return average_loss, accuracy


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device usado: {device}")

    transform = create_transform()
    train_dataset, val_dataset, test_dataset = create_datasets(transform)
    train_loader, val_loader, _test_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
    )

    model = create_model()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    trainable_parameters = (
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    optimizer = optim.Adam(
        trainable_parameters,
        lr=LEARNING_RATE,
    )

    best_val_accuracy = 0.0
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )
        val_loss, val_accuracy = validate(
            model,
            val_loader,
            criterion,
            device,
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), MODEL_PATH)

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Acc: {train_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Acc: {val_accuracy:.4f}")

    print(f"\nMelhor modelo salvo em: {MODEL_PATH}")


if __name__ == "__main__":
    main()
