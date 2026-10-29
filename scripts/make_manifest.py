"""Build and freeze artifacts/manifest.csv (file, source, label, split, phash).

Splits:
  PlantVillage (capped at PV_CAP per class): 80% train / 10% tune / 10% test_lab
  PlantWild:                                 70% train / 15% tune / 15% test_field
  PlantDoc (all admitted classes):           test_cross (never trained or tuned on)

Images are resized once into data/cache. Near-duplicates are grouped before splitting
(PlantWild), and any test image with a perceptual-hash near-twin in train/tune is dropped.
Refuses to overwrite an existing manifest unless --force is given.
"""
import argparse
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agrisense.config import (CACHE, CACHE_SHORT_SIDE, CLASSES, MANIFEST, PLANTDOC_MAP,  # noqa: E402
                              PLANTVILLAGE_MAP, PLANTWILD_MAP, PV_CAP, RAW, ROOT, SEED)

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
NEAR_DUP_BITS = 6  # Hamming distance on 64-bit pHash counted as a near-twin


def list_sources() -> pd.DataFrame:
    rows = []
    for d, label in PLANTVILLAGE_MAP.items():
        for p in sorted((RAW / "plantvillage/raw/color" / d).glob("*")):
            if p.suffix.lower() in IMG_EXT:
                rows.append(("plantvillage", label, p))
    for d, label in PLANTWILD_MAP.items():
        for p in sorted((RAW / "plantwild/plantwild/images" / d).glob("*")):
            if p.suffix.lower() in IMG_EXT:
                rows.append(("plantwild", label, p))
    for part in ("train", "test"):
        for d, label in PLANTDOC_MAP.items():
            for p in sorted((RAW / "plantdoc" / part / d).glob("*")):
                if p.suffix.lower() in IMG_EXT:
                    rows.append(("plantdoc", label, p))
    return pd.DataFrame(rows, columns=["source", "label", "orig"])


def cap_plantvillage(df: pd.DataFrame) -> pd.DataFrame:
    keep = []
    for label, g in df[df.source == "plantvillage"].groupby("label"):
        idx = list(g.index)
        random.Random(SEED).shuffle(idx)
        keep += idx[:PV_CAP]
    return pd.concat([df.loc[keep], df[df.source != "plantvillage"]])


def cache_one(args):
    i, source, label, orig = args
    out = CACHE / source / label / f"{i:06d}.jpg"
    try:
        if out.exists():
            im = Image.open(out).convert("RGB")
        else:
            im = ImageOps.exif_transpose(Image.open(orig)).convert("RGB")
            w, h = im.size
            s = CACHE_SHORT_SIDE / min(w, h)
            if s < 1:
                im = im.resize((round(w * s), round(h * s)), Image.BICUBIC)
            out.parent.mkdir(parents=True, exist_ok=True)
            im.save(out, quality=92)
        return i, out.relative_to(ROOT).as_posix(), int(str(imagehash.phash(im)), 16)
    except Exception as e:  # corrupt or unreadable file
        print("skip", orig, e)
        return i, None, None


def hamming_min(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """For each hash in a, the minimum Hamming distance to any hash in b."""
    out = np.empty(len(a), dtype=np.int64)
    for s in range(0, len(a), 256):
        x = a[s:s + 256, None] ^ b[None, :]
        out[s:s + 256] = np.bitwise_count(x).min(axis=1)
    return out


def group_split(df: pd.DataFrame, fracs: dict[str, float]) -> pd.Series:
    """Split per class, keeping exact-pHash duplicates in the same split."""
    split = pd.Series(index=df.index, dtype=object)
    for label, g in df.groupby("label"):
        groups = list(g.groupby("phash").groups.values())
        random.Random(f"{SEED}-{label}").shuffle(groups)
        n = sum(len(x) for x in groups)
        names, bounds = list(fracs), np.cumsum(list(fracs.values())) * n
        seen = 0
        for grp in groups:
            k = int(np.searchsorted(bounds, seen, side="right"))
            split[list(grp)] = names[min(k, len(names) - 1)]
            seen += len(grp)
    return split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if MANIFEST.exists() and not args.force:
        sys.exit(f"{MANIFEST} is frozen. Use --force only if you have not trained yet.")

    df = cap_plantvillage(list_sources()).reset_index(drop=True)
    print(df.groupby(["source", "label"]).size().unstack(0).fillna(0).astype(int))

    with ThreadPoolExecutor(12) as ex:
        res = list(ex.map(cache_one, [(i, r.source, r.label, r.orig) for i, r in df.iterrows()]))
    df["file"] = [f for _, f, _ in res]
    df["phash"] = pd.Series([h for _, _, h in res], index=df.index, dtype=object)
    df = df.dropna(subset=["file"]).copy()
    df["phash"] = np.array([int(h) for h in df.phash], dtype=np.uint64)

    df["split"] = None
    pv, pw = df.source == "plantvillage", df.source == "plantwild"
    df.loc[pv, "split"] = group_split(df[pv], {"train": .8, "tune": .1, "test_lab": .1})
    df.loc[pw, "split"] = group_split(df[pw], {"train": .7, "tune": .15, "test_field": .15})
    df.loc[df.source == "plantdoc", "split"] = "test_cross"

    # Exact duplicates inside PlantDoc (it ships train/test folders that overlap).
    pd_mask = df.source == "plantdoc"
    dup = df[pd_mask].duplicated(subset=["label", "phash"])
    print("plantdoc exact duplicates removed:", int(dup.sum()))
    df = df.drop(dup[dup].index)

    # Drop test images with a near-twin in train/tune.
    fit = df[df.split.isin(["train", "tune"])].phash.to_numpy()
    for s in ("test_lab", "test_field", "test_cross"):
        m = df.split == s
        d = hamming_min(df.loc[m, "phash"].to_numpy(), fit)
        leak = df[m].index[d <= NEAR_DUP_BITS]
        print(f"{s}: dropped {len(leak)} near-twins of training/tuning images")
        df = df.drop(leak)

    df["phash"] = df.phash.map(lambda h: f"{int(h):016x}")
    df["orig"] = df.orig.map(lambda p: Path(p).relative_to(ROOT).as_posix())
    df = df[["file", "source", "label", "split", "phash", "orig"]].sort_values(["split", "label", "file"])
    MANIFEST.parent.mkdir(exist_ok=True)
    df.to_csv(MANIFEST, index=False)

    counts = df.groupby(["label", "split"]).size().unstack(fill_value=0).reindex(CLASSES, fill_value=0)
    print(counts)
    field_train = df[(df.source != "plantdoc") & (df.source != "plantvillage")].groupby("label").size()
    for c in CLASSES:
        nf, nt = int(field_train.get(c, 0)), int(counts.loc[c].get("test_cross", 0))
        flag = "OK" if nf >= 60 and nt >= 15 else ("LAB-ONLY (no cross-source test)" if nt == 0 else "WARN")
        print(f"admission {c:28s} field={nf:4d} plantdoc_test={nt:4d} {flag}")
    print("wrote", MANIFEST, len(df))


if __name__ == "__main__":
    main()
