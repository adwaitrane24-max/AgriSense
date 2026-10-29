"""Save logits for every split (and corrupted copies of the field test sets) so that
calibration, thresholds and the report never need to re-run the model.

    python scripts/predict.py --variant baseline
    python scripts/predict.py --variant mixed --sweeps --aux

Writes runs/<variant>/<set>.npz with logits, y, files.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agrisense import model as M  # noqa: E402
from agrisense.config import AUX, CLASSES, RUNS  # noqa: E402
from agrisense.data import CORRUPTIONS, ManifestDataset, eval_tf, load_manifest  # noqa: E402

SPLITS = ["tune", "test_lab", "test_field", "test_cross"]


def run(model, ds, workers):
    out = []
    with torch.inference_mode():
        for x, _ in DataLoader(ds, batch_size=64, num_workers=workers):
            out.append(model(x).numpy())
    return np.concatenate(out) if out else np.zeros((0, len(CLASSES)), np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="mixed")
    ap.add_argument("--sweeps", action="store_true", help="brightness/blur/JPEG at 3 severities")
    ap.add_argument("--aux", action="store_true", help="non-leaf set and own photos")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    model, size = M.load(args.variant)
    df = load_manifest()
    out_dir = RUNS / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)

    def save(tag, d, corrupt=None):
        logits = run(model, ManifestDataset(d, eval_tf(size), corrupt), args.workers)
        np.savez(out_dir / f"{tag}.npz", logits=logits, y=d.y.to_numpy(), files=np.array(d.file.tolist()))
        acc = (logits.argmax(1) == d.y.to_numpy()).mean() if len(d) else float("nan")
        print(f"{args.variant} {tag:28s} n={len(d):4d} acc={acc:.3f}", flush=True)

    for s in SPLITS:
        save(s, df[df.split == s])
    if args.sweeps:
        for s in ("test_field", "test_cross"):
            for kind, levels in CORRUPTIONS.items():
                for lvl, fn in levels.items():
                    save(f"{s}__{kind}{lvl}", df[df.split == s], fn)
    if args.aux:
        for sub in ("nonleaf", "own"):
            files = sorted(p for p in (AUX / sub).glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
            if not files:
                continue
            d = pd.DataFrame({"file": [str(p) for p in files], "y": -1})
            save(sub, d)


if __name__ == "__main__":
    main()
