"""AgriSense: crop-leaf screening aid. Start with:  python app.py"""
import json
import time

import gradio as gr
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps

from agrisense import model as M
from agrisense.config import ARTIFACTS, CAUTION, CLASSES, MANIFEST, PRETTY, REPORT
from agrisense.data import eval_tf
from agrisense.trust import (decide_probs, leaf_fraction, load_cfg, name, quality_metrics,
                             quality_reason, softmax)

CFG = load_cfg()
try:
    MODEL, SIZE = M.load("mixed")
except FileNotFoundError:
    MODEL, SIZE = None, 224
TF = eval_tf(SIZE)
RELIABILITY = ARTIFACTS / "reliability.json"

COVERED = ", ".join(f"{c} {d.lower()}" for c, d in PRETTY.values())
TIPS = {
    "too blurry": "Hold the phone steady and tap the leaf on screen to focus.",
    "too dark": "Move into daylight or open shade; avoid using the photo at dusk.",
    "too bright": "Avoid direct midday sun and glare; shade the leaf with your body.",
}
PHOTO_TIPS = "Photo tips: one leaf filling most of the frame, in daylight or open shade, held steady, no flash."


def card(title: str, body: str, tone: str) -> str:
    return (f"<div class='res {tone}'><div class='res-title'>{title}</div><div>{body}</div>"
            f"<div class='caution'>{CAUTION}</div></div>")


def reliability_line(p: float) -> str:
    if not RELIABILITY.exists():
        return ""
    r = json.loads(RELIABILITY.read_text())
    return (f"<div class='rel'>When this tool says \"likely\" (about {r['mean_conf_pct']}% on average), "
            f"it was right about {r['n_of_10']} out of 10 times on field photos it had not seen "
            f"({r['n']} PlantDoc images).</div>")


def check(img: Image.Image | None):
    if img is None:
        return card("No photo yet", "Upload a photo or use the camera.", "neutral")
    t0 = time.time()
    img = ImageOps.exif_transpose(img).convert("RGB")
    reason = quality_reason(quality_metrics(img), CFG)
    if reason:
        return card("Retake", f"This photo is not clear enough to check: <b>{reason}</b>.<br>"
                    f"Tip: {TIPS[reason]} Fill the frame with one leaf.", "retake")
    outside = ("This is outside what this tool can check.<br><small>Covered: " + COVERED + ".</small>")
    if leaf_fraction(img) < CFG["leaf_min"] or MODEL is None:
        return card("Cannot assess", outside, "cannot")
    with torch.inference_mode():
        logits = MODEL(TF(img)[None]).numpy()[0]
    p = softmax(logits, CFG["temperature"])
    state, idx = decide_probs(p, CFG)
    ms = f"<div class='ms'>{(time.time() - t0) * 1000:.0f} ms</div>"
    if state == "likely":
        pct = int(round(p[idx[0]] * 100 / 5) * 5)
        return card("Likely", f"<b>{name(idx[0])}</b><br>about {pct}% confidence"
                    + reliability_line(p[idx[0]]) + ms, "likely")
    if state == "uncertain":
        a, b = idx
        return card("Uncertain", f"Not sure. This could be <b>{name(a)}</b> ({p[a] * 100:.0f}%) or "
                    f"<b>{name(b)}</b> ({p[b] * 100:.0f}%).<br>An expert should look at this leaf." + ms,
                    "uncertain")
    return card("Cannot assess", outside + ms, "cannot")


def covered_table() -> pd.DataFrame:
    rows = []
    df = pd.read_csv(MANIFEST) if MANIFEST.exists() else pd.DataFrame(columns=["label", "split", "source"])
    for c in CLASSES:
        d = df[df.label == c]
        tr = d[d.split == "train"]
        rows.append({"Crop": PRETTY[c][0], "Condition": PRETTY[c][1],
                     "Training images (lab)": int((tr.source == "plantvillage").sum()),
                     "Training images (field)": int((tr.source != "plantvillage").sum()),
                     "Held-out field test (PlantDoc)": int((d.split == "test_cross").sum())})
    return pd.DataFrame(rows)


def report_md() -> str:
    f = REPORT / "report.md"
    if not f.exists():
        return "Report not generated yet. Run `python scripts/report.py`."
    return f.read_text(encoding="utf-8").replace("](", "](/gradio_api/file=report/")


CSS = """
.res{border-radius:10px;padding:14px 16px;border:1px solid #ccc;font-size:1.05rem}
.res-title{font-weight:700;font-size:1.3rem;margin-bottom:6px}
.likely{border-color:#2e7d32;background:#2e7d3214}
.uncertain{border-color:#b26a00;background:#b26a0014}
.cannot{border-color:#555;background:#5551}
.retake{border-color:#c62828;background:#c6282814}
.caution{margin-top:12px;padding-top:8px;border-top:1px dashed #999;font-size:.85rem;opacity:.85}
.rel{margin-top:8px;font-size:.9rem}
.ms{font-size:.75rem;opacity:.5;margin-top:4px}
"""

with gr.Blocks(title="AgriSense leaf screening") as demo:
    gr.Markdown("# AgriSense: tomato and potato leaf screening")
    with gr.Tab("Check a leaf"):
        gr.Markdown(PHOTO_TIPS)
        with gr.Row():
            inp = gr.Image(type="pil", sources=["upload", "webcam"], label="Leaf photo")
            out = gr.HTML(card("No photo yet", "Upload a photo or use the camera.", "neutral"))
        inp.change(check, inp, out)
    with gr.Tab("Covered conditions"):
        gr.Markdown("**Crops:** tomato and potato. **Conditions:** listed below with the number of "
                    "images the model learned from.")
        gr.Dataframe(covered_table(), interactive=False)
        gr.Markdown("**Not covered:** other crops, pests, nutrient problems, several diseases on one "
                    "leaf, severity, whole plants, fruit. Anything else returns *cannot assess*. "
                    "Healthy potato has no images in the held-out PlantDoc test, so its field accuracy "
                    "is unknown.\n\n" + CAUTION)
    with gr.Tab("Report"):
        gr.Markdown(report_md())

if __name__ == "__main__":
    demo.launch(allowed_paths=[str(REPORT)], css=CSS)
