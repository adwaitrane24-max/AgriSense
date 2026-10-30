"""Turn uploaded bytes into the exact array shape the model expects.

Steps:
    1. validate_upload: reject wrong type, empty, too large or undecodable files.
    2. to_model_input: decode -> fix phone rotation -> RGB -> resize 224x224
       -> scale pixels from [0, 255] to [-1, 1] -> add batch dimension.

The scaling `x / 127.5 - 1` is exactly what
`tf.keras.applications.mobilenet_v2.preprocess_input` does. Writing it with
numpy keeps TensorFlow out of this module, so its tests run in milliseconds.

Complexity (p = number of pixels in the uploaded image):
    validate_upload  O(p) time (full decode check), O(p) space
    to_model_input   O(p) time to decode + resize, O(p) space while decoding;
                     the output is always 224*224*3 floats (~600 KB)
"""

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from leafcheck import config

# First bytes ("magic numbers") of real JPEG and PNG files.
_JPEG_MAGIC = b"\xff\xd8\xff"
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class InvalidImageError(ValueError):
    """Raised when an upload is not a usable JPG/PNG image."""


def validate_upload(
    filename: str, data: bytes, max_bytes: int = config.MAX_UPLOAD_BYTES
) -> None:
    """Check that an upload is a decodable JPG/PNG under the size limit.

    Args:
        filename: name sent by the browser (only the extension is used).
        data: raw file bytes.
        max_bytes: size limit in bytes.

    Raises:
        InvalidImageError: with a message that is safe to show the user.
    """
    if not data:
        raise InvalidImageError("The uploaded file is empty.")
    if len(data) > max_bytes:
        limit_mb = max_bytes / (1024 * 1024)
        raise InvalidImageError(f"File is too large. Maximum size is {limit_mb:.0f} MB.")

    extension = Path(filename or "").suffix.lower()
    if extension not in config.ALLOWED_EXTENSIONS:
        raise InvalidImageError("Only JPG and PNG images are accepted.")
    if not (data.startswith(_JPEG_MAGIC) or data.startswith(_PNG_MAGIC)):
        raise InvalidImageError("File content is not a JPG or PNG image.")

    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()  # checks the file structure without keeping pixels
    except (UnidentifiedImageError, OSError, SyntaxError) as error:
        raise InvalidImageError("The image could not be read. It may be corrupted.") from error


def to_model_input(data: bytes, size: int = config.IMG_SIZE) -> np.ndarray:
    """Decode image bytes into a (1, size, size, 3) float32 array in [-1, 1].

    Args:
        data: raw bytes of a JPG/PNG image (call validate_upload first).
        size: target width and height in pixels.

    Returns:
        A batch of one image, ready for model.predict().
    """
    with Image.open(io.BytesIO(data)) as image:
        image = ImageOps.exif_transpose(image)  # phone photos store rotation in EXIF
        image = image.convert("RGB")  # drops alpha / greyscale -> 3 channels
        image = image.resize((size, size), Image.Resampling.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32)

    pixels = pixels / 127.5 - 1.0
    return np.expand_dims(pixels, axis=0)
