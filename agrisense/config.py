"""Single source of truth for classes, label mappings, paths and seeds."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
CACHE = DATA / "cache"          # resized copies used for training/eval
AUX = DATA / "aux"              # quality test set, non-leaf set, own photos
ARTIFACTS = ROOT / "artifacts"  # model weights, trust.json, manifest
REPORT = ROOT / "report"
RUNS = ROOT / "runs"            # saved predictions (logits) per model and split

MANIFEST = ARTIFACTS / "manifest.csv"
TRUST = ARTIFACTS / "trust.json"

SEED = 42
IMG_SIZE = 224
CACHE_SHORT_SIDE = 288
PV_CAP = 500

# Class order is the model's output order. Do not reorder after training.
CLASSES = [
    "tomato_healthy",
    "tomato_early_blight",
    "tomato_late_blight",
    "tomato_septoria_leaf_spot",
    "tomato_bacterial_spot",
    "potato_healthy",
    "potato_early_blight",
    "potato_late_blight",
]

PRETTY = {
    "tomato_healthy": ("Tomato", "Healthy"),
    "tomato_early_blight": ("Tomato", "Early blight"),
    "tomato_late_blight": ("Tomato", "Late blight"),
    "tomato_septoria_leaf_spot": ("Tomato", "Septoria leaf spot"),
    "tomato_bacterial_spot": ("Tomato", "Bacterial spot"),
    "potato_healthy": ("Potato", "Healthy"),
    "potato_early_blight": ("Potato", "Early blight"),
    "potato_late_blight": ("Potato", "Late blight"),
}

# Source folder name -> class. Folders not listed are ignored.
PLANTVILLAGE_MAP = {
    "Tomato___healthy": "tomato_healthy",
    "Tomato___Early_blight": "tomato_early_blight",
    "Tomato___Late_blight": "tomato_late_blight",
    "Tomato___Septoria_leaf_spot": "tomato_septoria_leaf_spot",
    "Tomato___Bacterial_spot": "tomato_bacterial_spot",
    "Potato___healthy": "potato_healthy",
    "Potato___Early_blight": "potato_early_blight",
    "Potato___Late_blight": "potato_late_blight",
}

PLANTWILD_MAP = {
    "tomato leaf": "tomato_healthy",
    "tomato early blight": "tomato_early_blight",
    "tomato late blight": "tomato_late_blight",
    "tomato septoria leaf spot": "tomato_septoria_leaf_spot",
    "tomato bacterial leaf spot": "tomato_bacterial_spot",
    "potato leaf": "potato_healthy",
    "potato early blight": "potato_early_blight",
    "potato late blight": "potato_late_blight",
}

PLANTDOC_MAP = {
    "Tomato leaf": "tomato_healthy",
    "Tomato Early blight leaf": "tomato_early_blight",
    "Tomato leaf late blight": "tomato_late_blight",
    "Tomato Septoria leaf spot": "tomato_septoria_leaf_spot",
    "Tomato leaf bacterial spot": "tomato_bacterial_spot",
    "Potato leaf": "potato_healthy",  # PlantDoc may not have this folder
    "Potato leaf early blight": "potato_early_blight",
    "Potato leaf late blight": "potato_late_blight",
}

CAUTION = (
    "Preliminary screening aid. This is not a confirmed diagnosis or treatment "
    "advice. Please consult a local agriculture expert before acting."
)
