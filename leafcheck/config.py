"""All tunable settings in one place, so no other module hard-codes numbers."""

from pathlib import Path

# Folders
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
MODEL_DIR: Path = ROOT_DIR / "model"
MODEL_PATH: Path = MODEL_DIR / "leafcheck_mobilenetv2.keras"
LABELS_PATH: Path = MODEL_DIR / "labels.json"
SAMPLES_DIR: Path = ROOT_DIR / "samples"

# Model input
IMG_SIZE: int = 224  # MobileNetV2 expects 224 x 224 RGB images
TOP_K: int = 3  # how many classes to show on the result card

# Below this top confidence we say "Not sure" instead of naming a class
CONFIDENCE_THRESHOLD: float = 0.60

# Upload rules
MAX_UPLOAD_BYTES: int = 5 * 1024 * 1024  # 5 MB
ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})

# LRU cache size (number of distinct images remembered)
CACHE_CAPACITY: int = 128

# User-facing text
NOT_SURE_MESSAGE: str = "Not sure, please consult an agriculture expert."
DISCLAIMER: str = (
    "This is an automated guess for learning purposes. Confirm with a local "
    "agriculture expert or Krishi Vigyan Kendra before taking any action."
)
