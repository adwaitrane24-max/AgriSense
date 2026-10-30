"""Grad-CAM: show which parts of the leaf pushed the model towards a class.

Idea (Selvaraju et al., 2017): the last convolutional block of MobileNetV2
gives a 7 x 7 grid of 1280 feature maps. We ask "if this feature map got a
little stronger, how much would the score of the chosen class go up?" (the
gradient). Averaging that gradient over the grid gives one weight per
channel; the weighted sum of channels is a 7 x 7 heatmap of evidence.

Complexity:
    compute_heatmap  one extra forward pass + one backward pass through the
                     fixed network, so roughly 2x the cost of one prediction
                     (constant per image, because the input size is fixed).
                     Space O(7 * 7 * 1280) for features and gradients.
    overlay_png      O(pixels) time and space (224 * 224 resize, colour, blend,
                     PNG encode).
"""

import base64
import io

import numpy as np
from PIL import Image


def _find_base(model):  # type: ignore[no-untyped-def]
    """Return (base_model, head_layers): the nested keras.Model and layers after it."""
    import keras

    layers = model.layers
    for position, layer in enumerate(layers):
        if isinstance(layer, keras.Model):
            return layer, layers[position + 1:]
    raise ValueError("no nested base model found in the classifier")


def compute_heatmap(model, batch: np.ndarray, class_index: int) -> np.ndarray:  # type: ignore[no-untyped-def]
    """Grad-CAM heatmap for `class_index` on a (1, 224, 224, 3) batch.

    Steps: features = base(x); scores = head(features); gradient of the class
    score w.r.t. features; channel weights = mean gradient over the 7 x 7 grid;
    heatmap = ReLU(sum of weighted channels), scaled to [0, 1].

    Returns a 7 x 7 float32 array in [0, 1] (all zeros if there is no positive
    evidence). Cost: about 2x one prediction (forward + backward pass).
    """
    import tensorflow as tf

    base, head = _find_base(model)
    x = tf.convert_to_tensor(batch, dtype=tf.float32)
    with tf.GradientTape() as tape:
        features = base(x, training=False)
        tape.watch(features)
        out = features
        for layer in head:
            if isinstance(layer, tf.keras.layers.Dropout):
                continue  # dropout does nothing at inference time
            out = layer(out)
        score = out[:, class_index]
    grads = tape.gradient(score, features)  # (1, 7, 7, 1280)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))  # (1280,)
    cam = tf.reduce_sum(features[0] * weights, axis=-1)  # (7, 7)
    cam = tf.nn.relu(cam).numpy().astype(np.float32)
    peak = float(cam.max())
    return cam / peak if peak > 0 else cam


def _jet(values: np.ndarray) -> np.ndarray:
    """Map values in [0, 1] to RGB uint8 with a simple jet-like colormap. O(pixels)."""
    v = np.clip(values, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    green = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    blue = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return (np.stack([red, green, blue], axis=-1) * 255).astype(np.uint8)


def overlay_png(batch: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> str:
    """Blend a coloured heatmap over the input image; return a PNG data URL.

    `batch` is the (1, H, W, 3) model input in [-1, 1]; it is mapped back to
    RGB with (x + 1) * 127.5. The heatmap is resized to H x W with Pillow.
    Time and space O(pixels).
    """
    image = np.clip((batch[0] + 1.0) * 127.5, 0, 255).astype(np.uint8)
    height, width = image.shape[:2]
    heat = Image.fromarray((np.clip(heatmap, 0, 1) * 255).astype(np.uint8))
    heat = np.asarray(heat.resize((width, height), Image.BILINEAR), dtype=np.float32) / 255.0
    coloured = _jet(heat).astype(np.float32)
    blended = (1 - alpha) * image.astype(np.float32) + alpha * coloured
    buffer = io.BytesIO()
    Image.fromarray(blended.clip(0, 255).astype(np.uint8)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
