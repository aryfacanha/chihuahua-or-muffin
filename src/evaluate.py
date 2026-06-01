import argparse
import matplotlib
from pathlib import Path

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from config import DEFAULT_ARCHITECTURE, MODEL_PATH, PROJECT_ROOT, REPORTS_DIR
from dataset import create_dataloaders, create_datasets, create_transform
from device import describe_device, get_device
from model import SUPPORTED_ARCHITECTURES, create_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Avalia um checkpoint treinado no conjunto de teste.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODEL_PATH,
        help=f"Caminho do checkpoint do modelo. Padrão: {MODEL_PATH}",
    )
    parser.add_argument(
        "--architecture",
        default=DEFAULT_ARCHITECTURE,
        choices=sorted(SUPPORTED_ARCHITECTURES),
        help=f"Arquitetura usada pelo checkpoint. Padrão: {DEFAULT_ARCHITECTURE}",
    )

    return parser.parse_args()


def get_test_data():
    transform = create_transform()
    train_dataset, val_dataset, test_dataset = create_datasets(transform)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
    )

    return test_dataset, test_loader


def load_model(device, model_path, architecture=DEFAULT_ARCHITECTURE):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {model_path}. "
            "Treine um modelo ou informe --model-path com um checkpoint válido."
        )

    model = create_model(pretrained=False, architecture=architecture)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model


def collect_predictions(model, test_loader, device):
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1).cpu()

            all_labels.extend(labels.tolist())
            all_predictions.extend(predictions.tolist())

    return all_labels, all_predictions


def calculate_metrics(labels, predictions, class_names):
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average="weighted")
    recall = recall_score(labels, predictions, average="weighted")
    f1 = f1_score(labels, predictions, average="weighted")
    report = classification_report(
        labels,
        predictions,
        target_names=class_names,
    )
    matrix = confusion_matrix(labels, predictions)

    return accuracy, precision, recall, f1, report, matrix


def save_classification_report(report):
    report_path = REPORTS_DIR / "classification_report.txt"
    report_path.write_text(report, encoding="utf-8")


def save_confusion_matrix(matrix, class_names):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predição")
    plt.ylabel("Classe real")
    plt.title("Matriz de confusão")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png")
    plt.close()


def save_predictions(test_dataset, labels, predictions, class_names):
    image_paths = [image_path for image_path, _label in test_dataset.samples]
    rows = []

    for image_path, label, prediction in zip(image_paths, labels, predictions):
        relative_image_path = Path(image_path).relative_to(PROJECT_ROOT)
        rows.append({
            "image_path": relative_image_path.as_posix(),
            "true_label": class_names[label],
            "predicted_label": class_names[prediction],
        })

    predictions_df = pd.DataFrame(rows)
    predictions_df.to_csv(REPORTS_DIR / "predictions.csv", index=False)


def print_summary(accuracy, precision, recall, f1):
    print("Resumo da avaliação no conjunto de teste:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"\nResultados salvos em: {REPORTS_DIR}")


def main():
    args = parse_args()
    model_path = args.model_path.resolve()
    device = get_device()
    print(f"Device usado: {describe_device(device)}")
    print(f"Modelo usado: {model_path}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    test_dataset, test_loader = get_test_data()
    class_names = test_dataset.classes

    try:
        model = load_model(device, model_path, args.architecture)
    except FileNotFoundError as error:
        raise SystemExit(f"Erro: {error}") from None
    labels, predictions = collect_predictions(model, test_loader, device)

    accuracy, precision, recall, f1, report, matrix = calculate_metrics(
        labels,
        predictions,
        class_names,
    )

    save_classification_report(report)
    save_confusion_matrix(matrix, class_names)
    save_predictions(test_dataset, labels, predictions, class_names)
    print_summary(accuracy, precision, recall, f1)


if __name__ == "__main__":
    main()
