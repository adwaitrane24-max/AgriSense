"""Model wrapper plus the pure logic that turns probabilities into a result.

Pieces:
    pretty_label     "Tomato___Early_blight" -> ("Tomato", "Early blight")
    top_k            k most likely classes using a heap
    build_result     top-k + confidence threshold -> result dict for the API
    KerasPredictor   loads the trained MobileNetV2 once and predicts
    FakePredictor    same interface, no TensorFlow; used by tests and demo mode

Complexity (n = number of classes = 19, k = 3):
    top_k            O(n log k) time, O(k) space (heapq.nlargest keeps a heap of k)
    label lookup     O(1): labels is a list indexed by class id
    model predict    constant per image: fixed 224x224 input through a fixed
                     network (~300 million multiply-adds); model file ~10-14 MB
"""

import heapq
import json
from pathlib import Path
from typing import Any, Optional, Protocol, Sequence

import numpy as np

from leafcheck import config

_CROP_NAMES = {"Pepper,_bell": "Bell pepper", "Corn": "Corn (maize)"}


class Predictor(Protocol):
    """Anything with class labels that maps an image batch to probabilities."""

    labels: list[str]

    def predict_proba(self, batch: np.ndarray) -> np.ndarray:
        """Return a 1-D array of len(labels) probabilities for one image."""
        ...


def pretty_label(raw: str) -> tuple[str, str]:
    """Split a dataset class name into readable (crop, condition). O(len(raw)).

    Example: "Tomato___Tomato_mosaic_virus" -> ("Tomato", "Mosaic virus").
    """
    crop_raw, _, condition_raw = raw.partition("___")
    crop = _CROP_NAMES.get(crop_raw, crop_raw.replace("_", " "))
    condition = condition_raw.replace("_", " ").strip()
    if condition.lower().startswith("tomato "):  # "Tomato mosaic virus" -> "mosaic virus"
        condition = condition[len("tomato "):]
    condition = condition[:1].upper() + condition[1:].lower() if condition else ""
    return crop, condition


def top_k(probabilities: Sequence[float], k: int = config.TOP_K) -> list[int]:
    """Indices of the k largest probabilities, largest first.

    Uses heapq.nlargest, which keeps a min-heap of size k while scanning:
    O(n log k) time and O(k) extra space, versus O(n log n) for a full sort.
    """
    return heapq.nlargest(k, range(len(probabilities)), key=lambda i: probabilities[i])


def build_result(
    probabilities: Sequence[float],
    labels: Sequence[str],
    k: int = config.TOP_K,
    threshold: float = config.CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    """Turn raw model probabilities into the JSON-ready result.

    If the best confidence is below `threshold`, status is "not_sure" and the
    top-1 label is withheld (the top-k list is still shown for transparency).

    Time O(n log k), space O(k).
    """
    if len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must have the same length")

    top: list[dict[str, Any]] = []
    for index in top_k(probabilities, k):
        crop, condition = pretty_label(labels[index])
        top.append(
            {
                "label": labels[index],
                "crop": crop,
                "condition": condition,
                "healthy": condition.lower() == "healthy",
                "confidence": round(float(probabilities[index]), 4),
            }
        )

    best = top[0]
    confident = best["confidence"] >= threshold
    return {
        "status": "ok" if confident else "not_sure",
        "prediction": best if confident else None,
        "top": top,
        "message": None if confident else config.NOT_SURE_MESSAGE,
        "threshold": threshold,
        "disclaimer": config.DISCLAIMER,
    }


class KerasPredictor:
    """Loads the trained Keras model and labels once; reused for every request."""

    def __init__(
        self, model_path: Path = config.MODEL_PATH, labels_path: Path = config.LABELS_PATH
    ) -> None:
        """Load model and labels from disk. Slow (a few seconds), done once at startup."""
        import tensorflow as tf  # imported here so the rest of the app works without TF

        self.labels: list[str] = json.loads(Path(labels_path).read_text(encoding="utf-8"))
        self._model = tf.keras.models.load_model(model_path)
        outputs = int(self._model.output_shape[-1])
        if outputs != len(self.labels):
            raise ValueError(f"model has {outputs} outputs but labels.json has {len(self.labels)}")
        # TensorFlow's first call is ~2x slower (graph setup); do it now, not on the first upload.
        self.predict_proba(np.zeros((1, config.IMG_SIZE, config.IMG_SIZE, 3), dtype=np.float32))

    def predict_proba(self, batch: np.ndarray) -> np.ndarray:
        """Run the CNN on a (1, 224, 224, 3) batch; return 19 softmax probabilities."""
        return self._model(batch, training=False).numpy()[0]

    def heatmap(self, batch: np.ndarray, class_index: int) -> str | None:
        """Grad-CAM overlay for `class_index` as a PNG data URL, or None on failure.

        About 2x the cost of one prediction (extra forward + backward pass).
        Any error is swallowed so a heatmap problem never breaks prediction.
        """
        try:
            from leafcheck.gradcam import compute_heatmap, overlay_png

            return overlay_png(batch, compute_heatmap(self._model, batch, class_index))
        except Exception:  # noqa: BLE001 - heatmap is optional extra information
            return None


class FakePredictor:
    """Stand-in predictor with no TensorFlow, for tests and demo mode.

    If `probabilities` is given, it is always returned. Otherwise the output is
    derived from the image's average colour, so different images give different
    (but repeatable) answers. `calls` counts how often the "model" really ran,
    which lets tests prove that cache hits skip prediction.
    """

    def __init__(
        self,
        labels: Optional[Sequence[str]] = None,
        probabilities: Optional[Sequence[float]] = None,
    ) -> None:
        self.labels: list[str] = list(labels or config.CLASS_NAMES)
        self._fixed = None if probabilities is None else np.asarray(probabilities, dtype=float)
        self.calls = 0

    def predict_proba(self, batch: np.ndarray) -> np.ndarray:
        """Return fixed or colour-seeded pseudo probabilities summing to 1."""
        self.calls += 1
        if self._fixed is not None:
            return self._fixed
        seed = int((batch.mean() + 1.0) * 1_000_000) % (2**32)
        scores = np.random.default_rng(seed).random(len(self.labels)) ** 8
        return scores / scores.sum()

    def heatmap(self, batch: np.ndarray, class_index: int) -> str | None:
        """No real model, so no Grad-CAM heatmap. O(1)."""
        return None


def load_predictor() -> tuple[Predictor, bool]:
    """Return (predictor, is_demo). Uses the real model if both files exist."""
    if config.MODEL_PATH.exists() and config.LABELS_PATH.exists():
        return KerasPredictor(), False
    return FakePredictor(), True
