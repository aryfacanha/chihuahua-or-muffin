from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "kaggle"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
HARD_CASES_DATA_DIR = PROJECT_ROOT / "data" / "hard_cases"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
KAGGLE_DATASET_SLUG = "samuelcortinhas/muffin-vs-chihuahua-image-classification"

MODEL_PATH = MODELS_DIR / "best_model.pth"
MODEL_HISTORY_PATH = MODELS_DIR / "model_history.csv"
DEFAULT_ARCHITECTURE = "mobilenet_v2"

CLASSES = ["chihuahua", "muffin"]
SPLITS = ["train", "val", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 0
EPOCHS = 5
LEARNING_RATE = 0.001
IMAGE_SIZE = (224, 224)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
