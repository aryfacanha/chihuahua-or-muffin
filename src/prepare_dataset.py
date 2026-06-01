import argparse
import random
import shutil
from pathlib import Path

from config import (
    CLASSES,
    IMAGE_EXTENSIONS,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SEED,
    SPLITS,
)

DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepara o dataset em pastas de treino, validação e teste.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help=(
            "Caminho do dataset bruto. "
            f"Padrão: {RAW_DATA_DIR}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help=(
            "Caminho de saída do dataset processado. "
            f"Padrão: {PROCESSED_DATA_DIR}"
        ),
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help=f"Proporção do conjunto de treino. Padrão: {DEFAULT_TRAIN_RATIO}",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
        help=f"Proporção do conjunto de validação. Padrão: {DEFAULT_VAL_RATIO}",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=DEFAULT_TEST_RATIO,
        help=f"Proporção do conjunto de teste. Padrão: {DEFAULT_TEST_RATIO}",
    )

    return parser.parse_args()


def is_image(path):
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def validate_ratios(train_ratio, val_ratio, test_ratio):
    ratios = [train_ratio, val_ratio, test_ratio]

    if any(ratio <= 0 for ratio in ratios):
        raise ValueError("As proporções devem ser maiores que zero.")

    if not abs(sum(ratios) - 1.0) < 1e-9:
        raise ValueError("As proporções de treino, validação e teste devem somar 1.0.")


def find_class_dirs(raw_dir, class_name):
    return sorted(
        path
        for path in raw_dir.rglob(class_name)
        if path.is_dir() and path.name == class_name
    )


def find_images(raw_dir, class_name):
    images = []

    for class_dir in find_class_dirs(raw_dir, class_name):
        images.extend(image for image in class_dir.iterdir() if is_image(image))

    return sorted(images)


def validate_raw_dataset(raw_dir):
    if not raw_dir.exists():
        raise FileNotFoundError(f"Diretório bruto não encontrado: {raw_dir}")

    if not raw_dir.is_dir():
        raise NotADirectoryError(f"O caminho bruto não é um diretório: {raw_dir}")

    images_by_class = {}

    for class_name in CLASSES:
        class_dirs = find_class_dirs(raw_dir, class_name)

        if not class_dirs:
            raise FileNotFoundError(
                f"Nenhuma pasta da classe '{class_name}' encontrada em: {raw_dir}"
            )

        images = find_images(raw_dir, class_name)

        if not images:
            supported_extensions = ", ".join(sorted(IMAGE_EXTENSIONS))
            raise FileNotFoundError(
                "Nenhuma imagem suportada encontrada para a classe "
                f"'{class_name}'. Extensões suportadas: {supported_extensions}"
            )

        images_by_class[class_name] = images

    return images_by_class


def split_images(images, train_ratio, val_ratio, seed=SEED):
    random.seed(seed)
    shuffled_images = images[:]
    random.shuffle(shuffled_images)

    total = len(shuffled_images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    return {
        "train": shuffled_images[:train_end],
        "val": shuffled_images[train_end:val_end],
        "test": shuffled_images[val_end:],
    }


def prepare_folders(output_dir):
    for split in SPLITS:
        for class_name in CLASSES:
            target_dir = output_dir / split / class_name
            target_dir.mkdir(parents=True, exist_ok=True)

            for file_path in target_dir.iterdir():
                if is_image(file_path):
                    file_path.unlink()


def copy_images(output_dir, split_name, class_name, images):
    target_dir = output_dir / split_name / class_name

    for image_path in images:
        shutil.copy2(image_path, target_dir / image_path.name)


def print_summary(raw_dir, output_dir, ratios, counts):
    train_ratio, val_ratio, test_ratio = ratios

    print("Resumo da preparação do dataset")
    print(f"Dataset de origem: {raw_dir}")
    print(f"Dataset processado: {output_dir}")
    print(
        "Proporção da divisão: "
        f"train={train_ratio:.2f}, val={val_ratio:.2f}, test={test_ratio:.2f}"
    )
    print(f"Seed: {SEED}")
    print(f"Classes: {', '.join(CLASSES)}")
    print("\nQuantidade de imagens por split e classe:")

    for split in SPLITS:
        print(f"\n{split}:")
        for class_name in CLASSES:
            print(f"  {class_name}: {counts[split][class_name]}")


def main():
    args = parse_args()
    raw_dir = args.raw_dir.resolve()
    output_dir = args.output_dir.resolve()
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)

    try:
        validate_ratios(*ratios)
        images_by_class = validate_raw_dataset(raw_dir)
    except (FileNotFoundError, NotADirectoryError, ValueError) as error:
        raise SystemExit(f"Erro: {error}") from None

    prepare_folders(output_dir)

    counts = {
        split: {class_name: 0 for class_name in CLASSES}
        for split in SPLITS
    }

    for class_name, images in images_by_class.items():
        split_data = split_images(images, *ratios)

        for split_name, split_images_list in split_data.items():
            copy_images(output_dir, split_name, class_name, split_images_list)
            counts[split_name][class_name] = len(split_images_list)

    print_summary(raw_dir, output_dir, ratios, counts)


if __name__ == "__main__":
    main()
