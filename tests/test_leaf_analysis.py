"""Tests for the colour-based leaf analysis (no model needed)."""

import io
from pathlib import Path

import pytest
from PIL import Image, ImageFilter

from leafcheck.leaf_analysis import analyse_leaf, laplacian_variance, severity_band
from tests.helpers import make_image_bytes

SAMPLES = Path(__file__).resolve().parent.parent / "samples"
GREEN = (40, 160, 60)
BROWN = (140, 90, 40)


def _encode(image: Image.Image, fmt: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def _green_with_brown_block(size: int = 100, block: int = 50) -> bytes:
    """Green square with a brown block covering (block/size)^2 of the area."""
    image = Image.new("RGB", (size, size), GREEN)
    image.paste(BROWN, (0, 0, block, block))
    return _encode(image)


def _sample(name: str) -> bytes:
    return (SAMPLES / name).read_bytes()


def test_returns_expected_keys_and_types() -> None:
    result = analyse_leaf(make_image_bytes(color=GREEN))
    assert set(result) == {
        "leaf_percent",
        "affected_percent",
        "brightness",
        "sharpness",
        "warnings",
        "severity_band",
    }
    assert isinstance(result["leaf_percent"], float)
    assert isinstance(result["affected_percent"], float)
    assert isinstance(result["brightness"], int)
    assert isinstance(result["sharpness"], int)
    assert isinstance(result["warnings"], list)
    assert result["severity_band"] in {"none", "mild", "moderate", "severe"}


def test_pure_green_is_all_healthy_leaf() -> None:
    result = analyse_leaf(make_image_bytes(color=GREEN, fmt="PNG"))
    assert result["leaf_percent"] == pytest.approx(100.0)
    assert result["affected_percent"] == pytest.approx(0.0)
    assert result["severity_band"] == "none"


def test_brown_block_quarter_is_moderately_affected() -> None:
    result = analyse_leaf(_green_with_brown_block())
    assert 20 <= result["affected_percent"] <= 30
    assert result["severity_band"] == "moderate"
    assert result["leaf_percent"] > 95


def test_white_image_is_not_leaf() -> None:
    result = analyse_leaf(make_image_bytes(color=(255, 255, 255)))
    assert result["leaf_percent"] == 0.0
    assert result["affected_percent"] == 0.0
    assert "not_leaf" in result["warnings"]
    assert "too_bright" in result["warnings"]


def test_black_image_is_too_dark() -> None:
    result = analyse_leaf(make_image_bytes(color=(0, 0, 0)))
    assert "too_dark" in result["warnings"]
    assert "not_leaf" in result["warnings"]


def test_blurred_image_gets_blurry_warning() -> None:
    checker = Image.new("RGB", (200, 200), GREEN)
    for x in range(0, 200, 10):
        for y in range(0, 200, 10):
            if (x + y) // 10 % 2:
                checker.paste(BROWN, (x, y, x + 10, y + 10))
    blurred = checker.filter(ImageFilter.GaussianBlur(8))
    assert "blurry" not in analyse_leaf(_encode(checker))["warnings"]
    assert "blurry" in analyse_leaf(_encode(blurred))["warnings"]


def test_large_image_is_downscaled_but_same_percentages() -> None:
    small = analyse_leaf(_green_with_brown_block(100, 50))
    large = analyse_leaf(_green_with_brown_block(1000, 500))
    assert large["affected_percent"] == pytest.approx(small["affected_percent"], abs=1.0)


def test_laplacian_of_flat_image_is_zero() -> None:
    import numpy as np

    assert laplacian_variance(np.full((10, 10), 128.0)) == 0.0
    assert laplacian_variance(np.zeros((2, 2))) == 0.0


@pytest.mark.parametrize(
    "percent,band",
    [(0, "none"), (4.9, "none"), (5, "mild"), (14.9, "mild"), (15, "moderate"),
     (34.9, "moderate"), (35, "severe"), (100, "severe")],
)
def test_severity_band_boundaries(percent: float, band: str) -> None:
    assert severity_band(percent) == band


# --- Real sample images (calibration checks) ---------------------------------

HEALTHY = ["tomato_healthy.jpg", "potato_healthy.jpg"]
DISEASED = ["corn_common_rust.jpg", "tomato_early_blight.jpg", "pepper_bacterial_spot.jpg"]


@pytest.mark.parametrize("name", HEALTHY)
def test_healthy_samples_have_low_affected(name: str) -> None:
    result = analyse_leaf(_sample(name))
    assert result["affected_percent"] < 8
    assert "not_leaf" not in result["warnings"]


@pytest.mark.parametrize("name", DISEASED)
def test_diseased_samples_more_affected_than_healthy(name: str) -> None:
    worst_healthy = max(analyse_leaf(_sample(h))["affected_percent"] for h in HEALTHY)
    result = analyse_leaf(_sample(name))
    assert result["affected_percent"] > worst_healthy
    assert result["severity_band"] != "none"


def test_coffee_mug_is_not_leaf() -> None:
    result = analyse_leaf(_sample("not_a_leaf_coffee.jpg"))
    assert "not_leaf" in result["warnings"]
    assert result["leaf_percent"] < 5
