"""Tests for upload validation and model-input preprocessing."""

import numpy as np
import pytest

from leafcheck.preprocess import InvalidImageError, to_model_input, validate_upload
from tests.helpers import make_image_bytes


@pytest.mark.parametrize("width,height", [(50, 80), (224, 224), (1000, 600)])
def test_any_size_becomes_model_shape(width: int, height: int) -> None:
    array = to_model_input(make_image_bytes(width, height))
    assert array.shape == (1, 224, 224, 3)
    assert array.dtype == np.float32


def test_pixels_scaled_to_minus_one_one() -> None:
    white = to_model_input(make_image_bytes(color=(255, 255, 255)))
    black = to_model_input(make_image_bytes(color=(0, 0, 0)))
    assert np.allclose(white, 1.0)
    assert np.allclose(black, -1.0)


def test_rgba_png_becomes_three_channels() -> None:
    array = to_model_input(make_image_bytes(fmt="PNG", mode="RGBA"))
    assert array.shape[-1] == 3


def test_valid_jpeg_and_png_pass() -> None:
    validate_upload("leaf.jpg", make_image_bytes(fmt="JPEG"))
    validate_upload("leaf.PNG", make_image_bytes(fmt="PNG"))


def test_rejects_empty_file() -> None:
    with pytest.raises(InvalidImageError, match="empty"):
        validate_upload("leaf.jpg", b"")


def test_rejects_text_file() -> None:
    with pytest.raises(InvalidImageError):
        validate_upload("notes.txt", b"hello, this is not an image")


def test_rejects_text_renamed_as_jpg() -> None:
    with pytest.raises(InvalidImageError, match="not a JPG or PNG"):
        validate_upload("fake.jpg", b"hello, this is not an image")


def test_rejects_oversize_file() -> None:
    data = make_image_bytes()
    with pytest.raises(InvalidImageError, match="too large"):
        validate_upload("leaf.jpg", data, max_bytes=len(data) - 1)


def test_rejects_truncated_image() -> None:
    data = make_image_bytes(400, 400, fmt="PNG")
    with pytest.raises(InvalidImageError):
        validate_upload("leaf.png", data[:100])
