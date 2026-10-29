"""Fit everything the trust layer needs from the TUNING split only, then write artifacts/trust.json.

  * temperature T (minimise NLL of logits / T)
  * high threshold: lowest p1 cut-off giving >= 90% accuracy on accepted tuning images
  * set threshold: lowest p1+p2 cut-off so that, for images below the high threshold that pass
    it, the true class is in the top two >= 90% of the time
  * blur threshold: from clean tuning images vs. synthetically blurred copies

Using one tuning split for checkpoint choice and calibration is a deliberate simplification.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agrisense.config import ROOT, RUNS, SEED, TRUST  # noqa: E402
from agrisense.data import load_manifest  # noqa: E402
from agrisense.trust import DEFAULTS, quality_metrics, softmax  # noqa: E402

TARGET = 0.90
MIN_ACCEPTED = 30


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    z, t = torch.tensor(logits, dtype=torch.float64), torch.tensor(y)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(z / log_t.exp(), t)
        loss.backward()
        return loss
    opt.step(closure)
    return float(log_t.exp())


def pick_high(p: np.ndarray, y: np.ndarray) -> float:
    p1, correct = p.max(1), p.argmax(1) == y
    for t in np.arange(0.30, 0.995, 0.005):
        acc = p1 >= t
        if acc.sum() >= MIN_ACCEPTED and correct[acc].mean() >= TARGET:
            return round(float(t), 3)
    return 0.99


def pick_set(p: np.ndarray, y: np.ndarray, high: float) -> float:
    order = np.argsort(-p, 1)
    s = np.sort(p, 1)[:, ::-1]
    band = s[:, 0] < high
    top2 = (order[:, 0] == y) | (order[:, 1] == y)
    for t in np.arange(0.30, 0.995, 0.005):
        acc = band & (s[:, 0] + s[:, 1] >= t)
        if acc.sum() >= MIN_ACCEPTED and top2[acc].mean() >= TARGET:
            return round(float(t), 3)
    return 0.99


def pick_blur(df) -> dict:
    rng = np.random.default_rng(SEED)
    files = df.file.sample(min(200, len(df)), random_state=SEED).tolist()
    clean, blurred = [], []
    for f in files:
        im = Image.open(ROOT / f).convert("RGB")
        clean.append(quality_metrics(im)["blur"])
        blurred.append(quality_metrics(im.filter(ImageFilter.GaussianBlur(rng.uniform(2.5, 5))))["blur"])
    clean, blurred = np.array(clean), np.array(blurred)
    # Highest threshold that keeps false alarms on clean tuning images at or under 10%.
    thr = float(np.percentile(clean, 10))
    return {"blur_min": round(thr, 1),
            "blur_tune_false_alarm": float((clean < thr).mean()),
            "blur_tune_detect": float((blurred < thr).mean())}


def main():
    cfg = dict(DEFAULTS)
    df = load_manifest()
    tune = df[df.split == "tune"]
    for variant in ("baseline", "mixed"):
        f = RUNS / variant / "tune.npz"
        if not f.exists():
            continue
        d = np.load(f)
        mask = np.ones(len(d["y"]), bool)
        if variant == "baseline":  # baseline was tuned on PlantVillage only
            mask = np.char.find(d["files"].astype(str), "/plantvillage/") >= 0
        t = fit_temperature(d["logits"][mask], d["y"][mask])
        cfg[f"temperature_{variant}"] = round(t, 4)
        print(variant, "T =", round(t, 3))
    d = np.load(RUNS / "mixed" / "tune.npz")
    cfg["temperature"] = cfg["temperature_mixed"]
    p = softmax(d["logits"], cfg["temperature"])
    cfg["high"] = pick_high(p, d["y"])
    cfg["set"] = pick_set(p, d["y"], cfg["high"])
    cfg.update(pick_blur(tune))
    TRUST.write_text(json.dumps(cfg, indent=2))
    print(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()
