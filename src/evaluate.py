import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib

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

from config import (
    DEFAULT_ARCHITECTURE,
    MODEL_PATH,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
)
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPORTS_DIR / "evaluations",
        help="Diretório base para salvar o histórico de avaliações.",
    )

    return parser.parse_args()


def to_relative_path(path):
    path = Path(path).resolve()

    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def get_test_data():
    test_dir = PROCESSED_DATA_DIR / "test"

    if not test_dir.exists():
        raise FileNotFoundError(
            f"Dataset de teste não encontrado: {test_dir}. "
            "Prepare o dataset antes de avaliar."
        )

    transform = create_transform()
    train_dataset, val_dataset, test_dataset = create_datasets(transform)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
    )

    if len(test_dataset) == 0:
        raise ValueError(f"Dataset de teste vazio: {test_dir}")

    if not test_dataset.classes:
        raise ValueError(f"Nenhuma classe encontrada no dataset de teste: {test_dir}")

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


def build_evaluation_dir(model_path, output_dir, evaluated_at):
    timestamp = evaluated_at.strftime("%Y%m%d_%H%M%S")
    base_name = f"{model_path.stem}_{timestamp}"
    evaluation_dir = output_dir / base_name
    suffix = 1

    while evaluation_dir.exists():
        evaluation_dir = output_dir / f"{base_name}_{suffix}"
        suffix += 1

    evaluation_dir.mkdir(parents=True)

    return evaluation_dir


def save_classification_report(report, output_dir):
    report_path = output_dir / "classification_report.txt"
    report_path.write_text(report, encoding="utf-8")


def save_confusion_matrix(matrix, class_names, output_dir):
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
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()


def save_predictions(test_dataset, labels, predictions, class_names, output_dir):
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
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)


def build_metrics(
    model_path,
    evaluation_dir,
    evaluated_at,
    accuracy,
    precision,
    recall,
    f1,
    matrix,
    class_names,
):
    classification_report_path = evaluation_dir / "classification_report.txt"
    confusion_matrix_path = evaluation_dir / "confusion_matrix.png"
    predictions_path = evaluation_dir / "predictions.csv"
    metrics_path = evaluation_dir / "metrics.json"

    return {
        "model_path": to_relative_path(model_path),
        "model_name": model_path.name,
        "evaluation_dir": to_relative_path(evaluation_dir),
        "evaluated_at": evaluated_at.isoformat(timespec="seconds"),
        "test_dataset_path": to_relative_path(PROCESSED_DATA_DIR / "test"),
        "classification_report_path": to_relative_path(classification_report_path),
        "confusion_matrix_path": to_relative_path(confusion_matrix_path),
        "predictions_path": to_relative_path(predictions_path),
        "metrics_path": to_relative_path(metrics_path),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": matrix.tolist(),
        "class_names": class_names,
    }


def save_metrics_json(metrics, output_dir):
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def append_evaluation_history(metrics):
    history_path = REPORTS_DIR / "evaluation_history.csv"
    history_exists = history_path.exists()
    history_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "evaluation_dir",
        "evaluated_at",
        "model_path",
        "model_name",
        "test_dataset_path",
        "confusion_matrix_path",
        "metrics_path",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
    ]

    with history_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not history_exists:
            writer.writeheader()

        writer.writerow({
            "evaluation_dir": metrics["evaluation_dir"],
            "evaluated_at": metrics["evaluated_at"],
            "model_path": metrics["model_path"],
            "model_name": metrics["model_name"],
            "test_dataset_path": metrics["test_dataset_path"],
            "confusion_matrix_path": metrics["confusion_matrix_path"],
            "metrics_path": metrics["metrics_path"],
            "accuracy": f"{metrics['accuracy']:.4f}",
            "precision": f"{metrics['precision']:.4f}",
            "recall": f"{metrics['recall']:.4f}",
            "f1_score": f"{metrics['f1_score']:.4f}",
        })


def save_compatibility_reports(evaluation_dir):
    save_targets = {
        "classification_report.txt": REPORTS_DIR / "classification_report.txt",
        "confusion_matrix.png": REPORTS_DIR / "confusion_matrix.png",
        "predictions.csv": REPORTS_DIR / "predictions.csv",
    }

    for filename, target_path in save_targets.items():
        target_path.write_bytes((evaluation_dir / filename).read_bytes())


def get_model_evaluation_dir(output_dir, model_name):
    model_stem = Path(model_name).stem
    model_dir = output_dir / "by_model" / model_stem
    model_dir.mkdir(parents=True, exist_ok=True)

    return model_dir


def append_model_evaluation_history(metrics, output_dir):
    model_dir = get_model_evaluation_dir(output_dir, metrics["model_name"])
    history_path = model_dir / "evaluation_history.csv"
    history_exists = history_path.exists()
    fieldnames = [
        "evaluation_dir",
        "evaluated_at",
        "model_path",
        "model_name",
        "test_dataset_path",
        "confusion_matrix_path",
        "metrics_path",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
    ]

    with history_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not history_exists:
            writer.writeheader()

        writer.writerow({
            "evaluation_dir": metrics["evaluation_dir"],
            "evaluated_at": metrics["evaluated_at"],
            "model_path": metrics["model_path"],
            "model_name": metrics["model_name"],
            "test_dataset_path": metrics["test_dataset_path"],
            "confusion_matrix_path": metrics["confusion_matrix_path"],
            "metrics_path": metrics["metrics_path"],
            "accuracy": f"{metrics['accuracy']:.4f}",
            "precision": f"{metrics['precision']:.4f}",
            "recall": f"{metrics['recall']:.4f}",
            "f1_score": f"{metrics['f1_score']:.4f}",
        })


def save_model_confusion_matrix_copy(metrics, output_dir):
    model_dir = get_model_evaluation_dir(output_dir, metrics["model_name"])
    evaluation_name = Path(metrics["evaluation_dir"]).name
    source_path = PROJECT_ROOT / metrics["confusion_matrix_path"]
    target_path = model_dir / f"{evaluation_name}_confusion_matrix.png"

    shutil.copy2(source_path, target_path)


def evaluate_model(model_path, architecture, output_dir, device):
    evaluated_at = datetime.now()
    evaluation_dir = build_evaluation_dir(model_path, output_dir, evaluated_at)

    test_dataset, test_loader = get_test_data()
    class_names = test_dataset.classes

    model = load_model(device, model_path, architecture)
    labels, predictions = collect_predictions(model, test_loader, device)

    accuracy, precision, recall, f1, report, matrix = calculate_metrics(
        labels,
        predictions,
        class_names,
    )

    metrics = build_metrics(
        model_path,
        evaluation_dir,
        evaluated_at,
        accuracy,
        precision,
        recall,
        f1,
        matrix,
        class_names,
    )

    save_classification_report(report, evaluation_dir)
    save_confusion_matrix(matrix, class_names, evaluation_dir)
    save_predictions(test_dataset, labels, predictions, class_names, evaluation_dir)
    save_metrics_json(metrics, evaluation_dir)
    append_evaluation_history(metrics)
    append_model_evaluation_history(metrics, output_dir)
    save_model_confusion_matrix_copy(metrics, output_dir)
    save_compatibility_reports(evaluation_dir)

    return metrics


def print_summary(metrics):
    print("Resumo da avaliação no conjunto de teste:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1_score']:.4f}")
    print(f"\nAvaliação salva em: {metrics['evaluation_dir']}")
    print(f"Últimos resultados também atualizados em: {REPORTS_DIR}")


def main():
    args = parse_args()
    model_path = args.model_path.resolve()
    output_dir = args.output_dir.resolve()
    device = get_device()
    print(f"Device usado: {describe_device(device)}")
    print(f"Modelo usado: {model_path}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        metrics = evaluate_model(
            model_path,
            args.architecture,
            output_dir,
            device,
        )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Erro: {error}") from None

    print_summary(metrics)


if __name__ == "__main__":
    main()
