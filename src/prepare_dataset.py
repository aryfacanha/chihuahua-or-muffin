import random
import shutil

from config import (
    CLASSES,
    IMAGE_EXTENSIONS,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SEED,
    SPLITS,
)


def is_image(path):
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def find_images(class_name):
    images = []

    for path in RAW_DATA_DIR.rglob(class_name):
        if path.is_dir() and path.name == class_name:
            images.extend(image for image in path.iterdir() if is_image(image))

    return sorted(images)


def split_images(images):
    random.seed(SEED)
    shuffled_images = images[:]
    random.shuffle(shuffled_images)

    total = len(shuffled_images)
    train_end = int(total * 0.70)
    val_end = train_end + int(total * 0.15)

    return {
        "train": shuffled_images[:train_end],
        "val": shuffled_images[train_end:val_end],
        "test": shuffled_images[val_end:],
    }


def prepare_folders():
    for split in SPLITS:
        for class_name in CLASSES:
            target_dir = PROCESSED_DATA_DIR / split / class_name
            target_dir.mkdir(parents=True, exist_ok=True)

            for file_path in target_dir.iterdir():
                if is_image(file_path):
                    file_path.unlink()


def copy_images(split_name, class_name, images):
    target_dir = PROCESSED_DATA_DIR / split_name / class_name

    for image_path in images:
        shutil.copy2(image_path, target_dir / image_path.name)


def print_counts(counts):
    print("Quantidade de imagens por split e classe:")

    for split in SPLITS:
        print(f"\n{split}:")
        for class_name in CLASSES:
            print(f"  {class_name}: {counts[split][class_name]}")


def main():
    prepare_folders()

    counts = {
        split: {class_name: 0 for class_name in CLASSES}
        for split in SPLITS
    }

    for class_name in CLASSES:
        images = find_images(class_name)
        split_data = split_images(images)

        for split_name, split_images_list in split_data.items():
            copy_images(split_name, class_name, split_images_list)
            counts[split_name][class_name] = len(split_images_list)

    print_counts(counts)


if __name__ == "__main__":
    main()
