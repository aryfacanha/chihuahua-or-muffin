import torch
from torch import nn
from torchvision import models

from config import DEFAULT_ARCHITECTURE
from dataset import create_dataloaders, create_datasets, create_transform
from device import describe_device, get_device

SUPPORTED_ARCHITECTURES = {DEFAULT_ARCHITECTURE}


def create_model(
    num_classes=2,
    freeze_features=True,
    pretrained=True,
    architecture=DEFAULT_ARCHITECTURE,
    unfreeze_last_blocks=0,
):
    if architecture != DEFAULT_ARCHITECTURE:
        supported = ", ".join(sorted(SUPPORTED_ARCHITECTURES))
        raise ValueError(
            f"Arquitetura não suportada: {architecture}. "
            f"Arquiteturas suportadas: {supported}"
        )

    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)

    if freeze_features:
        for parameter in model.features.parameters():
            parameter.requires_grad = False

        if unfreeze_last_blocks > 0:
            for block in model.features[-unfreeze_last_blocks:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def count_trainable_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def main():
    device = get_device()

    transform = create_transform()
    train_dataset, val_dataset, test_dataset = create_datasets(transform)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
    )

    images, labels = next(iter(train_loader))
    images = images.to(device)

    model = create_model(num_classes=2, freeze_features=True)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(images)

    print(f"Device usado: {describe_device(device)}")
    print(f"Shape do batch de entrada: {images.shape}")
    print(f"Shape da saida do modelo: {outputs.shape}")
    print(f"Quantidade total de parametros: {count_parameters(model)}")
    print(
        "Quantidade de parametros treinaveis: "
        f"{count_trainable_parameters(model)}"
    )


if __name__ == "__main__":
    main()
