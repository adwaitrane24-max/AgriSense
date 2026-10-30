"""Small helpers shared by tests (no real dataset needed)."""

import io

from PIL import Image


def make_image_bytes(
    width: int = 64,
    height: int = 64,
    color: tuple[int, int, int] = (40, 160, 60),
    fmt: str = "JPEG",
    mode: str = "RGB",
) -> bytes:
    """Create a solid-colour image in memory and return its encoded bytes."""
    fill = color + (255,) if mode == "RGBA" else color
    image = Image.new(mode, (width, height), fill)
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()
