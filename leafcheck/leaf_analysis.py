"""Rough colour-based leaf analysis and photo quality checks (no ML).

This module looks at pixel colours only. It is a rough estimate to help the
user, NOT a diagnosis: shadows, soil, dry edges or yellow light can all look
"affected". The CNN in predictor.py is the actual classifier.

How it works:
    1. Decode the image (fix phone rotation, RGB) and shrink it so the longest
       side is at most 256 px (keeps every step fast and predictable).
    2. Convert to HSV. Pillow stores hue, saturation and value as 0-255,
       so hue 0-255 maps to 0-360 degrees.
    3. Classify every pixel with numpy boolean masks:
         background  = low saturation (white/grey/black), very dark/bright,
                       or hues no leaf has (blue, purple, pure red)
         healthy     = green hue (about 60-170 degrees)
         affected    = yellow / orange / brown / tan hue (about 7-59 degrees)
       leaf (plant tissue) = healthy OR affected.
    4. Quality checks: mean brightness and Laplacian variance (sharpness).

Complexity (p = pixels after downscaling, at most 256*256 = 65,536):
    analyse_leaf  O(p) time, O(p) space. Decoding the original upload is
    O(original pixels), done once.
"""

import io
from typing import Any

import numpy as np
from PIL import Image, ImageOps

MAX_SIDE = 256

# Pillow HSV units (0-255). Hue degrees = value * 360 / 255.
# Thresholds were calibrated on the images in samples/.
MIN_GREEN_SATURATION = 40  # greener than grey: healthy tissue
MIN_BROWN_SATURATION = 18  # blight lesions are dull grey-brown (sat ~20-30)
MIN_TISSUE_VALUE = 30  # darker than this is shadow / background
MAX_TISSUE_VALUE = 250  # brighter than this is glare / background
GREEN_HUE_MIN = 42  # ~60 degrees (yellow-green)
GREEN_HUE_MAX = 120  # ~170 degrees (blue-green)
BROWN_HUE_MIN = 5  # ~7 degrees; pure reds (red objects, flowers) are excluded
# Brown/yellow range is BROWN_HUE_MIN <= hue < GREEN_HUE_MIN (~7-59 degrees).

DARK_BRIGHTNESS = 25
BRIGHT_BRIGHTNESS = 92
BLUR_SHARPNESS = 30  # calibrated on samples/; lower means blurry
NOT_LEAF_PERCENT = 15


def _load_rgb(data: bytes, max_side: int = MAX_SIDE) -> Image.Image:
    """Decode bytes to an upright RGB image no larger than max_side. O(p)."""
    with Image.open(io.BytesIO(data)) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
        return image.copy()


def _colour_masks(hsv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (healthy, affected) boolean masks for an HSV array. O(p)."""
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lit = (val >= MIN_TISSUE_VALUE) & (val <= MAX_TISSUE_VALUE)
    healthy = (
        lit
        & (sat >= MIN_GREEN_SATURATION)
        & (hue >= GREEN_HUE_MIN)
        & (hue <= GREEN_HUE_MAX)
    )
    affected = (
        lit
        & (sat >= MIN_BROWN_SATURATION)
        & (hue >= BROWN_HUE_MIN)
        & (hue < GREEN_HUE_MIN)
    )
    return healthy, affected


def laplacian_variance(grey: np.ndarray) -> float:
    """Variance of the 3x3 Laplacian [[0,1,0],[1,-4,1],[0,1,0]]. O(p).

    Sharp edges give large responses, so a blurry photo has a low variance.
    Implemented with array slicing (a convolution without border pixels).
    """
    if grey.shape[0] < 3 or grey.shape[1] < 3:
        return 0.0
    centre = grey[1:-1, 1:-1]
    lap = (
        grey[:-2, 1:-1] + grey[2:, 1:-1] + grey[1:-1, :-2] + grey[1:-1, 2:]
        - 4.0 * centre
    )
    return float(lap.var())


def severity_band(affected_percent: float) -> str:
    """Map an affected percentage to none / mild / moderate / severe. O(1)."""
    if affected_percent < 5:
        return "none"
    if affected_percent < 15:
        return "mild"
    if affected_percent < 35:
        return "moderate"
    return "severe"


def _warnings(brightness: float, sharpness: float, leaf_percent: float) -> list[str]:
    """Build the list of photo-quality warning codes. O(1)."""
    warnings: list[str] = []
    if brightness < DARK_BRIGHTNESS:
        warnings.append("too_dark")
    if brightness > BRIGHT_BRIGHTNESS:
        warnings.append("too_bright")
    if sharpness < BLUR_SHARPNESS:
        warnings.append("blurry")
    if leaf_percent < NOT_LEAF_PERCENT:
        warnings.append("not_leaf")
    return warnings


def analyse_leaf(data: bytes) -> dict[str, Any]:
    """Estimate leaf coverage, discoloured area and photo quality from colours.

    This is a rough colour-based estimate, not a diagnosis.

    Args:
        data: raw bytes of a JPG/PNG image (validate the upload first).

    Returns:
        dict with keys:
            leaf_percent (float): % of pixels that are plant tissue.
            affected_percent (float): % of leaf pixels that are yellow/brown.
            brightness (int): mean HSV value, 0-100.
            sharpness (int): Laplacian variance of the greyscale image.
            warnings (list[str]): too_dark, too_bright, blurry, not_leaf.
            severity_band (str): none, mild, moderate or severe.

    Complexity: O(p) time and O(p) space, p = pixels after downscaling.
    """
    image = _load_rgb(data)
    hsv = np.asarray(image.convert("HSV"), dtype=np.int16)
    grey = np.asarray(image.convert("L"), dtype=np.float32)

    healthy, affected = _colour_masks(hsv)
    total = hsv.shape[0] * hsv.shape[1]
    leaf_count = int(healthy.sum()) + int(affected.sum())

    leaf_percent = round(100.0 * leaf_count / total, 1) if total else 0.0
    affected_percent = (
        round(100.0 * int(affected.sum()) / leaf_count, 1) if leaf_count else 0.0
    )
    brightness = round(float(hsv[..., 2].mean()) * 100.0 / 255.0)
    sharpness = round(laplacian_variance(grey))

    return {
        "leaf_percent": leaf_percent,
        "affected_percent": affected_percent,
        "brightness": brightness,
        "sharpness": sharpness,
        "warnings": _warnings(brightness, sharpness, leaf_percent),
        "severity_band": severity_band(affected_percent),
    }
