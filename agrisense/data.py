"""Manifest-backed dataset, transforms and the class/domain-balanced sampler."""
import io

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms as T

from .config import CLASSES, IMG_SIZE, MANIFEST, ROOT

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
FIELD_SOURCES = {"plantwild", "plantdoc"}


def load_manifest() -> pd.DataFrame:
    df = pd.read_csv(MANIFEST)
    df["y"] = df.label.map(CLASSES.index)
    return df


class RandomJPEG:
    def __init__(self, p=0.3, q=(15, 80)):
        self.p, self.q = p, q

    def __call__(self, im):
        if np.random.rand() > self.p:
            return im
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=int(np.random.randint(*self.q)))
        return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


class RandomBlur:
    def __init__(self, p=0.2, sigma=(0.3, 2.0)):
        self.p, self.sigma = p, sigma

    def __call__(self, im):
        if np.random.rand() > self.p:
            return im
        return im.filter(ImageFilter.GaussianBlur(np.random.uniform(*self.sigma)))


def train_tf(size=IMG_SIZE):
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.5, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(20),
        T.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.2, hue=0.02),
        RandomBlur(),
        RandomJPEG(),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
        T.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])


def eval_tf(size=IMG_SIZE):
    return T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor(), T.Normalize(MEAN, STD)])


class ManifestDataset(Dataset):
    """`corrupt` is an optional PIL->PIL function applied before the transform (robustness sweeps)."""

    def __init__(self, df: pd.DataFrame, tf, corrupt=None):
        self.files = df.file.tolist()
        self.y = df.y.to_numpy()
        self.tf, self.corrupt = tf, corrupt

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        im = Image.open(ROOT / self.files[i]).convert("RGB")
        if self.corrupt is not None:
            im = self.corrupt(im)
        return self.tf(im), int(self.y[i])


def balanced_sampler(df: pd.DataFrame, num_samples: int) -> WeightedRandomSampler:
    """Every (class, lab/field) cell gets equal mass, so classes are balanced and field images
    make up about half of each epoch whenever field data exists for that class."""
    domain = df.source.isin(FIELD_SOURCES).map({True: "field", False: "lab"})
    cell = df.label + "|" + domain
    w = 1.0 / cell.map(cell.value_counts())
    return WeightedRandomSampler(torch.as_tensor(w.to_numpy(), dtype=torch.double), num_samples,
                                 replacement=True, generator=torch.Generator().manual_seed(0))


# Corruptions used by the robustness sweep (3 severities each).
def _brightness(f):
    return lambda im: ImageEnhance.Brightness(im).enhance(f)


def _blur(s):
    return lambda im: im.filter(ImageFilter.GaussianBlur(s))


def _jpeg(q):
    def f(im):
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=q)
        return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return f


CORRUPTIONS = {
    "dark": {1: _brightness(0.6), 2: _brightness(0.4), 3: _brightness(0.25)},
    "bright": {1: _brightness(1.4), 2: _brightness(1.8), 3: _brightness(2.4)},
    "blur": {1: _blur(1.0), 2: _blur(2.0), 3: _blur(4.0)},
    "jpeg": {1: _jpeg(30), 2: _jpeg(15), 3: _jpeg(5)},
}
