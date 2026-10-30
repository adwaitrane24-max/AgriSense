"""Tests for the Grad-CAM heatmap. Only the last test needs the real model."""

import base64
import io

import numpy as np
import pytest
from PIL import Image

from leafcheck import config
from leafcheck.gradcam import overlay_png
from leafcheck.predictor import FakePredictor

PREFIX = "data:image/png;base64,"


def _decode(data_url: str) -> Image.Image:
    assert data_url.startswith(PREFIX)
    return Image.open(io.BytesIO(base64.b64decode(data_url[len(PREFIX):])))


def test_overlay_png_returns_224_png_data_url() -> None:
    batch = np.zeros((1, 224, 224, 3), dtype=np.float32)
    heatmap = np.linspace(0, 1, 49, dtype=np.float32).reshape(7, 7)
    image = _decode(overlay_png(batch, heatmap))
    assert image.format == "PNG"
    assert image.size == (224, 224)


def test_overlay_png_handles_all_zero_heatmap() -> None:
    batch = np.full((1, 224, 224, 3), -1.0, dtype=np.float32)
    image = _decode(overlay_png(batch, np.zeros((7, 7), dtype=np.float32)))
    assert image.size == (224, 224)


def test_fake_predictor_has_no_heatmap() -> None:
    batch = np.zeros((1, 224, 224, 3), dtype=np.float32)
    assert FakePredictor().heatmap(batch, 0) is None


@pytest.mark.skipif(not config.MODEL_PATH.exists(), reason="trained model file missing")
def test_real_model_heatmap_on_sample() -> None:
    pytest.importorskip("tensorflow")
    from leafcheck.gradcam import compute_heatmap
    from leafcheck.predictor import KerasPredictor
    from leafcheck.preprocess import to_model_input

    data = (config.ROOT_DIR / "samples" / "tomato_early_blight.jpg").read_bytes()
    batch = to_model_input(data)
    predictor = KerasPredictor()
    class_index = int(np.argmax(predictor.predict_proba(batch)))

    heat = compute_heatmap(predictor._model, batch, class_index)
    assert heat.min() >= 0.0 and heat.max() <= 1.0
    assert heat.max() > 0.0

    url = predictor.heatmap(batch, class_index)
    assert url is not None
    assert _decode(url).size == (224, 224)
