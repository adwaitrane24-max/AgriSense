import timm
import torch

from .config import ARTIFACTS, CLASSES, IMG_SIZE


def build(pretrained=True) -> torch.nn.Module:
    return timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=len(CLASSES))


def save(model, variant: str, meta: dict) -> None:
    torch.save({"state_dict": model.state_dict(), "classes": CLASSES, **meta}, ARTIFACTS / f"{variant}.pt")


def load(variant: str = "mixed"):
    ck = torch.load(ARTIFACTS / f"{variant}.pt", map_location="cpu", weights_only=False)
    assert ck["classes"] == CLASSES, "class order changed since training"
    m = build(pretrained=False)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck.get("img_size", IMG_SIZE)
