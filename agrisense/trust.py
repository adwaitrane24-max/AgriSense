"""Trust layer: quality gate -> leaf guard -> calibrated prediction -> decision rule.

The same functions are used by the evaluation scripts and the app, so the report measures
exactly what the app does.
"""
import json

import cv2
import numpy as np
from PIL import Image

from .config import CLASSES, PRETTY, TRUST

QUALITY_SIDE = 256  # metrics are computed after resizing the short side to this

DEFAULTS = {
    "temperature": 1.0,
    "blur_min": 60.0,          # variance of the Laplacian; overwritten by calibrate.py
    "bright_min": 50.0,
    "bright_max": 205.0,
    "clip_max": 0.20,
    "leaf_min": 0.15,
    "high": 0.80,
    "set": 0.80,
}


def load_cfg() -> dict:
    cfg = dict(DEFAULTS)
    if TRUST.exists():
        cfg.update(json.loads(TRUST.read_text()))
    return cfg


def _small(im: Image.Image) -> np.ndarray:
    w, h = im.size
    s = QUALITY_SIDE / min(w, h)
    im = im.convert("RGB").resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    return np.asarray(im)


def quality_metrics(im: Image.Image) -> dict:
    rgb = _small(im)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return {
        "blur": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "brightness": float(gray.mean()),
        "clip_dark": float((gray <= 5).mean()),
        "clip_bright": float((gray >= 250).mean()),
    }


def quality_reason(m: dict, cfg: dict) -> str | None:
    """None if the photo passes, else 'too dark' / 'too bright' / 'too blurry'."""
    if m["brightness"] < cfg["bright_min"] or m["clip_dark"] > cfg["clip_max"]:
        return "too dark"
    if m["brightness"] > cfg["bright_max"] or m["clip_bright"] > cfg["clip_max"]:
        return "too bright"
    if m["blur"] < cfg["blur_min"]:
        return "too blurry"
    return None


def leaf_fraction(im: Image.Image) -> float:
    """Share of pixels with leaf-like colour: green, or yellow/brown lesion tones."""
    hsv = cv2.cvtColor(_small(im), cv2.COLOR_RGB2HSV)  # H in [0,180)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    green = (h >= 25) & (h <= 95) & (s >= 40) & (v >= 30)
    brown = (h >= 5) & (h < 25) & (s >= 60) & (v >= 30) & (v <= 220)
    return float((green | brown).mean())


def softmax(logits: np.ndarray, t: float = 1.0) -> np.ndarray:
    z = logits / t
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def decide_probs(p: np.ndarray, cfg: dict) -> tuple[str, list[int]]:
    """Return ('likely'|'uncertain'|'cannot_assess', top-2 class indices) for one prob vector."""
    order = np.argsort(-p)
    p1, p2 = p[order[0]], p[order[1]]
    if p1 >= cfg["high"]:
        return "likely", [int(order[0])]
    if p1 + p2 >= cfg["set"]:
        return "uncertain", [int(order[0]), int(order[1])]
    return "cannot_assess", []


def decide_batch(probs: np.ndarray, cfg: dict) -> np.ndarray:
    """Vectorised decision: 2 = likely, 1 = uncertain, 0 = cannot assess."""
    s = np.sort(probs, 1)[:, ::-1]
    return np.where(s[:, 0] >= cfg["high"], 2, np.where(s[:, 0] + s[:, 1] >= cfg["set"], 1, 0))


def name(i: int) -> str:
    crop, cond = PRETTY[CLASSES[i]]
    return f"{crop} - {cond.lower()}" if cond != "Healthy" else f"{crop} - healthy leaf"
