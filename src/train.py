import argparse
import copy
import csv
import shutil
from datetime import datetime

import torch
from torch import nn, optim

from config import (
    DEFAULT_ARCHITECTURE,
    EPOCHS,
    LEARNING_RATE,
    MODEL_HISTORY_PATH,
    MODEL_PATH,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
)
from dataset import create_dataloaders, create_datasets, create_transform
from device import describe_device, get_device
from model import create_model
from model import SUPPORTED_ARCHITECTURES


def parse_args():
    parser = argparse.ArgumentParser(
        description="Treina o classificador Chihuahua or Muffin.",
    )
    parser.add_argument(
        "--architecture",
        default=DEFAULT_ARCHITECTURE,
        choices=sorted(SUPPORTED_ARCHITECTURES),
        help=f"Arquitetura do modelo. Padrão: {DEFAULT_ARCHITECTURE}",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Observações opcionais para registrar no histórico de modelos.",
    )

    return parser.parse_args()


def build_checkpoint_path(architecture, created_at, val_accuracy):
    timestamp = created_at.strftime("%Y%m%d_%H%M%S")
    filename = (
        "chihuahua_muffin_"
        f"{architecture}_{timestamp}_valacc_{val_accuracy:.4f}.pth"
    )

    return MODEL_PATH.parent / filename


def append_model_history(
    model_path,
    architecture,
    created_at,
    epochs,
    best_val_accuracy,
    train_dataset_path,
    observations,
):
    history_exists = MODEL_HISTORY_PATH.exists()
    MODEL_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    relative_model_path = model_path.relative_to(PROJECT_ROOT)
    relative_train_dataset_path = train_dataset_path.relative_to(PROJECT_ROOT)

    with MODEL_HISTORY_PATH.open("a", newline="", encoding="utf-8") as file:
        fieldnames = [
            "model_path",
            "architecture",
            "created_at",
            "epochs",
            "best_val_accuracy",
            "train_dataset_path",
            "observations",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not history_exists:
            writer.writeheader()

        writer.writerow({
            "model_path": relative_model_path.as_posix(),
            "architecture": architecture,
            "created_at": created_at.isoformat(timespec="seconds"),
            "epochs": epochs,
            "best_val_accuracy": f"{best_val_accuracy:.4f}",
            "train_dataset_path": relative_train_dataset_path.as_posix(),
            "observations": observations,
        })


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
    args = parse_args()
    device = get_device()
    print(f"Device usado: {describe_device(device)}")

    transform = create_transform()
    train_dataset, val_dataset, test_dataset = create_datasets(transform)
    train_loader, val_loader, _test_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
    )

    architecture = args.architecture
    model = create_model(architecture=architecture)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    trainable_parameters = (
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    optimizer = optim.Adam(
        trainable_parameters,
        lr=LEARNING_RATE,
    )

    best_val_accuracy = -1.0
    best_state_dict = None
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
            best_state_dict = copy.deepcopy(model.state_dict())
            torch.save(best_state_dict, MODEL_PATH)

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Acc: {train_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Acc: {val_accuracy:.4f}")

    if best_state_dict is None:
        raise RuntimeError("Nenhum checkpoint foi gerado durante o treinamento.")

    created_at = datetime.now()
    checkpoint_path = build_checkpoint_path(
        architecture,
        created_at,
        best_val_accuracy,
    )
    torch.save(best_state_dict, checkpoint_path)
    shutil.copy2(checkpoint_path, MODEL_PATH)
    observations = args.notes or "Checkpoint copiado para models/best_model.pth"
    append_model_history(
        checkpoint_path,
        architecture,
        created_at,
        EPOCHS,
        best_val_accuracy,
        PROCESSED_DATA_DIR / "train",
        observations,
    )

    print(f"\nMelhor modelo salvo em: {checkpoint_path}")
    print(f"Cópia de compatibilidade atualizada em: {MODEL_PATH}")


if __name__ == "__main__":
    main()
