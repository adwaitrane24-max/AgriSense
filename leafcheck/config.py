"""All tunable settings in one place, so no other module hard-codes numbers."""

from pathlib import Path

# Folders
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
MODEL_DIR: Path = ROOT_DIR / "model"
MODEL_PATH: Path = MODEL_DIR / "leafcheck_mobilenetv2.keras"
LABELS_PATH: Path = MODEL_DIR / "labels.json"
SAMPLES_DIR: Path = ROOT_DIR / "samples"

# Demo images offered in the gallery on the upload page (files in samples/).
GALLERY_SAMPLES: tuple[str, ...] = (
    "tomato_early_blight.jpg",
    "potato_late_blight.jpg",
    "corn_common_rust.jpg",
    "pepper_bacterial_spot.jpg",
    "tomato_healthy.jpg",
    "potato_healthy.jpg",
    "field_potato_late_blight.jpg",
    "not_a_leaf_coffee.jpg",
)

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

# The 19 PlantVillage classes we keep, spelled exactly as TensorFlow Datasets
# names them. The training notebook uses this same list; the real order used by
# the model is saved in model/labels.json.
CLASS_NAMES: tuple[str, ...] = (
    "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
)

# User-facing text
NOT_SURE_MESSAGE: str = "Not sure, please consult an agriculture expert."
DISCLAIMER: str = (
    "This is an automated guess for learning purposes. Confirm with a local "
    "agriculture expert or Krishi Vigyan Kendra before taking any action."
)
