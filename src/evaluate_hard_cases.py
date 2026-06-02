import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix

from config import (
    CLASSES,
    DEFAULT_ARCHITECTURE,
    IMAGE_EXTENSIONS,
    MODEL_PATH,
    PROJECT_ROOT,
    REPORTS_DIR,
)
from device import describe_device, get_device
from model import SUPPORTED_ARCHITECTURES
from predict import load_model, predict_image


HARD_CASES_TEST_DIR = PROJECT_ROOT / "data" / "hard_cases" / "test"
HARD_CASES_REPORTS_DIR = REPORTS_DIR / "hard_cases"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Avalia imagens ambíguas separadas em data/hard_cases/test.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODEL_PATH,
        help=f"Caminho do checkpoint do modelo. Padrão: {MODEL_PATH}",
    )
    parser.add_argument(
        "--hard-cases-dir",
        type=Path,
        default=HARD_CASES_TEST_DIR,
        help=f"Diretório dos hard cases. Padrão: {HARD_CASES_TEST_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HARD_CASES_REPORTS_DIR,
        help=f"Diretório dos relatórios. Padrão: {HARD_CASES_REPORTS_DIR}",
    )
    parser.add_argument(
        "--architecture",
        default=DEFAULT_ARCHITECTURE,
        choices=sorted(SUPPORTED_ARCHITECTURES),
        help=f"Arquitetura usada pelo checkpoint. Padrão: {DEFAULT_ARCHITECTURE}",
    )

    return parser.parse_args()


def is_image(path):
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def to_relative_path(path):
    path = Path(path).resolve()

    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def collect_hard_case_images(hard_cases_dir):
    if not hard_cases_dir.exists():
        raise FileNotFoundError(f"Diretório de hard cases não encontrado: {hard_cases_dir}")

    samples = []

    for class_name in CLASSES:
        class_dir = hard_cases_dir / class_name

        if not class_dir.exists():
            raise FileNotFoundError(f"Pasta da classe não encontrada: {class_dir}")

        samples.extend(
            (image_path, class_name)
            for image_path in sorted(class_dir.iterdir())
            if is_image(image_path)
        )

    if not samples:
        raise ValueError(f"Nenhuma imagem encontrada em: {hard_cases_dir}")

    return samples


def evaluate_images(model, samples, device):
    rows = []

    for image_path, true_label in samples:
        image = Image.open(image_path).convert("RGB")
        predicted_index, confidence, _probabilities = predict_image(model, image, device)
        predicted_label = CLASSES[predicted_index]

        rows.append({
            "image_path": to_relative_path(image_path),
            "true_label": true_label,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "correct": predicted_label == true_label,
        })

    return rows


def save_predictions(rows, output_dir):
    predictions_path = output_dir / "hard_cases_predictions.csv"
    predictions_df = pd.DataFrame(rows)
    predictions_df.to_csv(predictions_path, index=False)

    return predictions_df


def save_confusion_matrix(matrix, output_dir):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Reds",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
    )
    plt.xlabel("Predição")
    plt.ylabel("Classe real")
    plt.title("Matriz de confusão - hard cases")
    plt.tight_layout()
    plt.savefig(output_dir / "hard_cases_confusion_matrix.png")
    plt.close()


def save_metrics(metrics, output_dir):
    metrics_path = output_dir / "hard_cases_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def calculate_error_counts(predictions_df):
    errors_df = predictions_df[predictions_df["correct"] == False]

    return {
        class_name: int((errors_df["true_label"] == class_name).sum())
        for class_name in CLASSES
    }


def build_metrics(predictions_df, matrix, model_path, hard_cases_dir):
    accuracy = accuracy_score(
        predictions_df["true_label"],
        predictions_df["predicted_label"],
    )

    return {
        "model_path": to_relative_path(model_path),
        "hard_cases_dir": to_relative_path(hard_cases_dir),
        "total_images": int(len(predictions_df)),
        "accuracy": accuracy,
        "errors_by_class": calculate_error_counts(predictions_df),
        "confusion_matrix": matrix.tolist(),
        "class_names": CLASSES,
    }


def print_summary(predictions_df, metrics):
    errors_df = predictions_df[predictions_df["correct"] == False]
    confident_errors_df = errors_df.sort_values(
        by="confidence",
        ascending=False,
    ).head(5)

    print("Resumo dos hard cases")
    print(f"Total de imagens: {metrics['total_images']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("Erros por classe:")

    for class_name, error_count in metrics["errors_by_class"].items():
        print(f"  {class_name}: {error_count}")

    if confident_errors_df.empty:
        print("\nNenhum erro encontrado.")
        return

    print("\nErros mais confiantes:")

    for row in confident_errors_df.to_dict("records"):
        print(
            f"  {row['image_path']} | real={row['true_label']} | "
            f"predito={row['predicted_label']} | confiança={row['confidence']:.4f}"
        )


def main():
    args = parse_args()
    model_path = args.model_path.resolve()
    hard_cases_dir = args.hard_cases_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Device usado: {describe_device(device)}")
    print(f"Modelo usado: {model_path}")
    print(f"Hard cases: {hard_cases_dir}")

    try:
        samples = collect_hard_case_images(hard_cases_dir)
        model = load_model(device, model_path, args.architecture)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Erro: {error}") from None

    rows = evaluate_images(model, samples, device)
    predictions_df = save_predictions(rows, output_dir)
    matrix = confusion_matrix(
        predictions_df["true_label"],
        predictions_df["predicted_label"],
        labels=CLASSES,
    )
    metrics = build_metrics(predictions_df, matrix, model_path, hard_cases_dir)

    save_confusion_matrix(matrix, output_dir)
    save_metrics(metrics, output_dir)
    print_summary(predictions_df, metrics)
    print(f"\nResultados salvos em: {output_dir}")


if __name__ == "__main__":
    main()
