"""Produce report/report.md and its PNG charts from saved predictions in runs/.

Every number comes from the frozen manifest + runs/*.npz + artifacts/trust.json. Hand-written
text lives in report/notes/*.md and report/notes/failure_notes.json and is pulled in verbatim.

    python scripts/report.py
"""
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageEnhance, ImageFilter  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agrisense import model as M  # noqa: E402
from agrisense.config import ARTIFACTS, AUX, CLASSES, PRETTY, REPORT, ROOT, RUNS, SEED  # noqa: E402
from agrisense.data import CORRUPTIONS, eval_tf, load_manifest  # noqa: E402
from agrisense.trust import (decide_batch, leaf_fraction, load_cfg, quality_metrics,  # noqa: E402
                             quality_reason, softmax)

NOTES = REPORT / "notes"
SETS = {"test_lab": "In-domain lab (PlantVillage)", "test_field": "In-domain field (PlantWild)",
        "test_cross": "Cross-source field (PlantDoc)"}
SHORT = {c: f"{PRETTY[c][0][:3]}. {PRETTY[c][1].lower()}" for c in CLASSES}
rng = np.random.default_rng(SEED)
CFG = load_cfg()


def load(variant, tag):
    f = RUNS / variant / f"{tag}.npz"
    return np.load(f) if f.exists() else None


def boot_ci(y, p, fn, n=1000):
    idx = rng.integers(0, len(y), (n, len(y)))
    vals = [fn(y[i], p[i]) for i in idx]
    return np.percentile(vals, [2.5, 97.5])


def macro_f1(y, p):
    return f1_score(y, p, labels=np.unique(y), average="macro", zero_division=0)


def acc_fn(y, p):
    return (y == p).mean()


def ece(prob, y, bins=15):
    conf, pred = prob.max(1), prob.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return e


def nll(prob, y):
    return float(-np.log(np.clip(prob[np.arange(len(y)), y], 1e-12, 1)).mean())


def pct(x):
    return f"{100 * x:.1f}%"


def fmt_ci(v, ci):
    return f"{100 * v:.1f}% [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]"


def savefig(name):
    plt.tight_layout()
    plt.savefig(REPORT / name, dpi=110)
    plt.close()
    return name


# ---------------------------------------------------------------- sections

def main_table(out):
    out.append("## 1. Accuracy and macro-F1\n")
    out.append("Bootstrap 95% intervals in brackets (1,000 resamples). Macro-F1 averages over the classes "
               "present in that test set. **The headline number is the cross-source PlantDoc row.**\n")
    out.append("| Model | Test set | n | Accuracy | Macro-F1 |\n|---|---|---|---|---|")
    res = {}
    for v in ("baseline", "mixed"):
        for s, title in SETS.items():
            d = load(v, s)
            if d is None:
                continue
            y, p = d["y"], d["logits"].argmax(1)
            a, f = acc_fn(y, p), macro_f1(y, p)
            res[(v, s)] = (a, f)
            out.append(f"| {v} | {title} | {len(y)} | {fmt_ci(a, boot_ci(y, p, acc_fn))} | "
                       f"{fmt_ci(f, boot_ci(y, p, macro_f1))} |")
    if ("baseline", "test_cross") in res and ("mixed", "test_cross") in res:
        gain = 100 * (res[("mixed", "test_cross")][1] - res[("baseline", "test_cross")][1])
        verdict = "**met**" if gain >= 10 else "**not met**"
        out.append(f"\nTarget 1 (mixed beats baseline by >= 10 macro-F1 points on PlantDoc): "
                   f"{gain:+.1f} points, {verdict}.\n")
    return res


def per_class(out):
    out.append("## 2. Per-class errors (mixed model)\n")
    for s in ("test_field", "test_cross"):
        d = load("mixed", s)
        if d is None:
            continue
        y, p = d["y"], d["logits"].argmax(1)
        pr, rc, _, sup = precision_recall_fscore_support(y, p, labels=range(len(CLASSES)), zero_division=0)
        out.append(f"**{SETS[s]}**\n\n| Class | n | Precision | Recall |\n|---|---|---|---|")
        for i, c in enumerate(CLASSES):
            if sup[i] == 0:
                out.append(f"| {PRETTY[c][0]} {PRETTY[c][1].lower()} | 0 | - | - (no test images) |")
                continue
            warn = " ⚠" if sup[i] < 15 else ""
            out.append(f"| {PRETTY[c][0]} {PRETTY[c][1].lower()}{warn} | {sup[i]} | {pct(pr[i])} | {pct(rc[i])} |")
        cm = confusion_matrix(y, p, labels=range(len(CLASSES)))
        fig, ax = plt.subplots(figsize=(6.4, 5.4))
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                            color="white" if norm[i, j] > .5 else "black")
        ax.set_xticks(range(len(CLASSES)), [SHORT[c] for c in CLASSES], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(CLASSES)), [SHORT[c] for c in CLASSES], fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion matrix, {SETS[s]}", fontsize=10)
        img = savefig(f"confusion_{s}.png")
        off = [(cm[i, j], i, j) for i in range(len(CLASSES)) for j in range(len(CLASSES)) if i != j and cm[i, j]]
        off.sort(reverse=True)
        out.append(f"\n![confusion]({img})\n\nTop confusions:")
        for n, i, j in off[:3]:
            out.append(f"- {PRETTY[CLASSES[i]][0]} {PRETTY[CLASSES[i]][1].lower()} is mistaken for "
                       f"{PRETTY[CLASSES[j]][0].lower()} {PRETTY[CLASSES[j]][1].lower()} "
                       f"{n} times out of {cm[i].sum()}.")
        out.append("")


def calibration(out):
    out.append("## 3. Calibration (temperature scaling)\n")
    out.append(f"Temperature fitted on the tuning split: T = {CFG['temperature']:.3f} (mixed), "
               f"{CFG.get('temperature_baseline', float('nan')):.3f} (baseline).\n")
    out.append("| Test set | n | ECE before | ECE after | NLL before | NLL after |\n|---|---|---|---|---|---|")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    stats = {}
    for ax, s in zip(axes, ("test_field", "test_cross")):
        d = load("mixed", s)
        if d is None:
            continue
        y = d["y"]
        p0, p1 = softmax(d["logits"]), softmax(d["logits"], CFG["temperature"])
        stats[s] = (ece(p0, y), ece(p1, y))
        out.append(f"| {SETS[s]} | {len(y)} | {ece(p0, y):.3f} | {ece(p1, y):.3f} | {nll(p0, y):.3f} | {nll(p1, y):.3f} |")
        for prob, lab in ((p0, "before"), (p1, "after")):
            conf, ok = prob.max(1), prob.argmax(1) == y
            edges = np.linspace(0, 1, 11)
            xs, ys = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (conf > lo) & (conf <= hi)
                if m.sum() >= 5:
                    xs.append(conf[m].mean())
                    ys.append(ok[m].mean())
            ax.plot(xs, ys, "o-", label=lab)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_title(SETS[s], fontsize=10)
        ax.set_xlabel("Stated confidence")
        ax.set_ylabel("Observed accuracy")
        ax.legend()
    img = savefig("reliability.png")
    if "test_field" in stats:
        e = stats["test_field"][1]
        out.append(f"\nTarget 2 (ECE <= 0.07 on in-domain field after scaling): {e:.3f}, "
                   f"**{'met' if e <= 0.07 else 'not met'}**.")
    out.append(f"\n![reliability]({img})\n")


def risk_coverage(out):
    out.append("## 4. Abstention: risk-coverage and result states\n")
    out.append(f"Thresholds chosen on the tuning split: **likely** if top probability >= {CFG['high']:.3f}; "
               f"**uncertain** if top-two sum >= {CFG['set']:.3f}; otherwise **cannot assess**.\n")
    fig, ax = plt.subplots(figsize=(5.5, 4))
    out.append("| Test set | n | Likely | Accuracy when likely | Uncertain | True class in top 2 when uncertain | Cannot assess |")
    out.append("|---|---|---|---|---|---|---|")
    summary = {}
    for s in SETS:
        d = load("mixed", s)
        if d is None:
            continue
        y, prob = d["y"], softmax(d["logits"], CFG["temperature"])
        conf, ok = prob.max(1), prob.argmax(1) == y
        order = np.argsort(-conf)
        cov = np.arange(1, len(y) + 1) / len(y)
        accs = np.cumsum(ok[order]) / np.arange(1, len(y) + 1)
        ax.plot(cov, accs, label=SETS[s])
        st = decide_batch(prob, CFG)
        top2 = np.argsort(-prob, 1)[:, :2]
        in2 = (top2 == y[:, None]).any(1)
        L, U, C = st == 2, st == 1, st == 0
        out.append(f"| {SETS[s]} | {len(y)} | {pct(L.mean())} | {pct(ok[L].mean()) if L.any() else '-'} | "
                   f"{pct(U.mean())} | {pct(in2[U].mean()) if U.any() else '-'} | {pct(C.mean())} |")
        summary[s] = {"cov": L.mean(), "acc": ok[L].mean() if L.any() else np.nan,
                      "acc_at_40": accs[max(0, int(0.4 * len(y)) - 1)]}
    ax.axvline(0.4, color="grey", ls=":", lw=1)
    ax.axhline(0.9, color="grey", ls=":", lw=1)
    ax.set_xlabel("Coverage (share of images answered)")
    ax.set_ylabel("Accuracy on answered images")
    ax.set_ylim(0.3, 1.01)
    ax.legend(fontsize=8)
    ax.set_title("Risk-coverage, mixed model + temperature", fontsize=10)
    img = savefig("risk_coverage.png")
    if "test_field" in summary:
        a = summary["test_field"]["acc_at_40"]
        out.append(f"\nTarget 3 (>= 90% accuracy at >= 40% coverage, in-domain field): accuracy at 40% "
                   f"coverage is {pct(a)}, **{'met' if a >= .9 else 'not met'}**.")
    out.append(f"\n![risk-coverage]({img})\n")
    return summary


def robustness(out):
    out.append("## 5. Lighting and image-quality robustness (model only, gate switched off)\n")
    out.append("Accuracy of the mixed model when every test image is degraded. Levels: darken x0.6/x0.4/x0.25, "
               "brighten x1.4/x1.8/x2.4, Gaussian blur sigma 1/2/4 px (on ~288 px images), JPEG quality 30/15/5.\n")
    out.append("| Test set | Clean | " + " | ".join(f"{k} {l}" for k in CORRUPTIONS for l in (1, 2, 3)) + " |")
    out.append("|---|---|" + "---|" * 12)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), sharey=True)
    for s in ("test_field", "test_cross"):
        clean = load("mixed", s)
        if clean is None or load("mixed", f"{s}__dark1") is None:
            continue
        a0 = acc_fn(clean["y"], clean["logits"].argmax(1))
        row = [pct(a0)]
        for ax, k in zip(axes, CORRUPTIONS):
            vals = [a0]
            for lvl in (1, 2, 3):
                d = load("mixed", f"{s}__{k}{lvl}")
                vals.append(acc_fn(d["y"], d["logits"].argmax(1)))
                row.append(pct(vals[-1]))
            ax.plot(range(4), vals, "o-", label=SETS[s])
            ax.set_title(k)
            ax.set_xticks(range(4), ["clean", "1", "2", "3"])
        out.append(f"| {SETS[s]} | " + " | ".join(row) + " |")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(fontsize=7)
    img = savefig("robustness.png")
    out.append(f"\n![robustness]({img})\n")


def quality_gate(out, df):
    out.append("## 6. Quality gate\n")
    test = df[df.split.isin(["test_field", "test_cross"])].sample(frac=1, random_state=SEED)
    bad_src, good_src = test.iloc[:100], test.iloc[100:300]
    kinds = ["blur", "dark", "glare"]
    r = np.random.default_rng(SEED)
    bad_flags, kinds_used = [], []
    for i, f in enumerate(bad_src.file):
        im = Image.open(ROOT / f).convert("RGB")
        k = kinds[i % 3]
        if k == "blur":
            im = im.filter(ImageFilter.GaussianBlur(r.uniform(3, 6)))
        elif k == "dark":
            im = ImageEnhance.Brightness(im).enhance(r.uniform(0.12, 0.3))
        else:
            im = ImageEnhance.Brightness(im).enhance(r.uniform(2.6, 3.5))
        bad_flags.append(quality_reason(quality_metrics(im), CFG) is not None)
        kinds_used.append(k)
    good_flags = [quality_reason(quality_metrics(Image.open(ROOT / f).convert("RGB")), CFG) is not None
                  for f in good_src.file]
    bad_flags, kinds_used = np.array(bad_flags), np.array(kinds_used)
    det, fa = bad_flags.mean(), np.mean(good_flags)
    out.append(f"Quality test set: 100 held-out test images degraded with strong blur, darkness or glare "
               f"(synthetic), plus 200 untouched test images. Blur threshold {CFG['blur_min']:.1f} "
               f"(variance of the Laplacian at 256 px) set on tuning images; brightness 50-205 and clipping 20% "
               f"fixed in advance.\n")
    out.append("| Bad-photo type | n | Flagged |\n|---|---|---|")
    for k in kinds:
        m = kinds_used == k
        out.append(f"| {k} | {m.sum()} | {pct(bad_flags[m].mean())} |")
    out.append(f"| **all bad** | 100 | **{pct(det)}** |\n| good test images (false alarms) | 200 | **{pct(fa)}** |")
    ok = det >= .8 and fa <= .2
    out.append(f"\nTarget 4 (>= 80% of bad photos flagged, <= 20% false alarms): **{'met' if ok else 'not met'}**. "
               "These bad photos are synthetic; real phone blur and glare are less uniform.\n")


def leaf_guard(out, df):
    out.append("## 7. Out-of-scope images (leaf guard)\n")
    nl = sorted((AUX / "nonleaf").glob("*.jpg"))
    if not nl:
        out.append("Non-leaf set not downloaded.\n")
        return
    leaves = df[df.split.isin(["test_field", "test_cross"])].sample(300, random_state=SEED)
    lf_leaf = np.array([leaf_fraction(Image.open(ROOT / f)) for f in leaves.file])
    lf_non = np.array([leaf_fraction(Image.open(p)) for p in nl])
    yb = np.r_[np.ones(len(lf_leaf)), np.zeros(len(lf_non))]
    auc_lf = roc_auc_score(yb, np.r_[lf_leaf, lf_non])
    line = (f"Non-leaf set: {len(nl)} random COCO photos (people, hands, food, fruit, objects, street scenes). "
            f"Leaf set: 300 test leaves.\n\n| Score | AUROC |\n|---|---|\n| Leaf-colour fraction | {auc_lf:.3f} |")
    nld = load("mixed", "nonleaf")
    end_to_end = ""
    if nld is not None:
        mx_non = softmax(nld["logits"], CFG["temperature"]).max(1)
        idx = {f: i for i, f in enumerate(np.concatenate([load("mixed", s)["files"] for s in ("test_field", "test_cross")]))}
        allp = np.concatenate([softmax(load("mixed", s)["logits"], CFG["temperature"]) for s in ("test_field", "test_cross")])
        mx_leaf = allp[[idx[f] for f in leaves.file]].max(1)
        auc_mx = roc_auc_score(yb, np.r_[mx_leaf, mx_non])
        comb = lambda lf, mx: np.where(lf < CFG["leaf_min"], 0, mx)  # noqa: E731
        auc_c = roc_auc_score(yb, np.r_[comb(lf_leaf, mx_leaf), comb(lf_non, mx_non)])
        line += f"\n| Max calibrated probability | {auc_mx:.3f} |\n| Leaf guard, then max probability | {auc_c:.3f} |"
        st_non = np.where(lf_non < CFG["leaf_min"], 0, decide_batch(softmax(nld["logits"], CFG["temperature"]), CFG))
        end_to_end = (f"\n\nEnd to end, {pct((st_non == 0).mean())} of non-leaf photos end as *cannot assess*, "
                      f"{pct((st_non == 1).mean())} as *uncertain* and {pct((st_non == 2).mean())} as *likely* "
                      f"(the last two are failures). The leaf guard alone rejects {pct((lf_leaf < CFG['leaf_min']).mean())} "
                      f"of real test leaves.")
        auc_lf = max(auc_lf, auc_c)
    out.append(line + end_to_end)
    out.append(f"\nTarget 5 (AUROC >= 0.9): best score {auc_lf:.3f}, **{'met' if auc_lf >= .9 else 'not met'}**. "
               "Leaves of crops we do not cover are *not* separated by this guard; they rely on the confidence "
               "thresholds (see discussion).\n")


def ablation(out):
    out.append("## 8. Ablation on PlantDoc (each row adds one step)\n")
    b, m = load("baseline", "test_cross"), load("mixed", "test_cross")
    if b is None or m is None:
        out.append("Missing predictions.\n")
        return
    y = m["y"]
    pb = softmax(b["logits"], CFG.get("temperature_baseline", 1.0))
    pm0, pm1 = softmax(m["logits"]), softmax(m["logits"], CFG["temperature"])
    st = decide_batch(pm1, CFG)
    L = st == 2
    ok = pm1.argmax(1) == y
    out.append("| Row | Accuracy | Macro-F1 | ECE | Answered (likely) | Accuracy on answered |\n|---|---|---|---|---|---|")
    out.append(f"| Baseline (PlantVillage only) | {pct(acc_fn(y, b['logits'].argmax(1)))} | "
               f"{pct(macro_f1(y, b['logits'].argmax(1)))} | {ece(softmax(b['logits']), y):.3f} | 100% | same |")
    out.append(f"| + mixed lab and field data | {pct(ok.mean())} | {pct(macro_f1(y, pm0.argmax(1)))} | "
               f"{ece(pm0, y):.3f} | 100% | same |")
    out.append(f"| + temperature scaling | {pct(ok.mean())} | {pct(macro_f1(y, pm1.argmax(1)))} | "
               f"{ece(pm1, y):.3f} | 100% | same |")
    out.append(f"| + abstention (likely only) | - | - | - | {pct(L.mean())} | "
               f"{pct(ok[L].mean()) if L.any() else '-'} |")
    out.append(f"\n(Baseline ECE shown before its own temperature; after it: {ece(pb, y):.3f}.)\n")
    rel = {"n": int(L.sum()), "mean_conf_pct": int(round(100 * pm1.max(1)[L].mean())) if L.any() else 0,
           "n_of_10": int(round(10 * ok[L].mean())) if L.any() else 0, "acc": float(ok[L].mean()) if L.any() else 0}
    (ARTIFACTS / "reliability.json").write_text(json.dumps(rel, indent=2))
    out.append(f"Reliability sentence used in the app: when the tool says *likely* (about {rel['mean_conf_pct']}% on "
               f"average), it was right about **{rel['n_of_10']} out of 10** times on {rel['n']} PlantDoc photos.\n")


def failures(out):
    out.append("## 9. Ten real failure cases (PlantDoc, mixed model)\n")
    m = load("mixed", "test_cross")
    if m is None:
        return
    prob = softmax(m["logits"], CFG["temperature"])
    pred, conf, y = prob.argmax(1), prob.max(1), m["y"]
    wrong = np.where(pred != y)[0]
    pick = wrong[np.argsort(-conf[wrong])][:10]
    notes_f = NOTES / "failure_notes.json"
    notes = json.loads(notes_f.read_text()) if notes_f.exists() else {}
    fig, axes = plt.subplots(2, 5, figsize=(13, 6))
    out.append("| # | File | True | Predicted (confidence) | App outcome | Likely reason |\n|---|---|---|---|---|---|")
    st = decide_batch(prob, CFG)
    for k, (ax, i) in enumerate(zip(axes.flat, pick), 1):
        f = str(m["files"][i])
        ax.imshow(Image.open(ROOT / f).convert("RGB"))
        ax.set_title(f"#{k} true: {SHORT[CLASSES[y[i]]]}\npred: {SHORT[CLASSES[pred[i]]]} {conf[i]:.0%}", fontsize=8)
        ax.axis("off")
        outcome = ["cannot assess", "uncertain", "likely"][st[i]]
        out.append(f"| {k} | `{Path(f).name}` | {SHORT[CLASSES[y[i]]]} | {SHORT[CLASSES[pred[i]]]} ({conf[i]:.0%}) | "
                   f"{outcome} | {notes.get(Path(f).name, 'to be annotated')} |")
    for ax in list(axes.flat)[len(pick):]:
        ax.axis("off")
    img = savefig("failures.png")
    out.append(f"\n![failures]({img})\n")


def latency(out):
    try:
        model, size = M.load("mixed")
    except FileNotFoundError:
        return
    tf = eval_tf(size)
    im = Image.new("RGB", (1200, 900), (60, 140, 50))
    times = []
    for _ in range(12):
        t0 = time.time()
        quality_metrics(im)
        leaf_fraction(im)
        with torch.inference_mode():
            model(tf(im)[None])
        times.append(time.time() - t0)
    med = float(np.median(times[2:]))
    out.append(f"## 10. Speed\n\nMedian end-to-end time per photo (gate + guard + model) on the laptop CPU: "
               f"**{med * 1000:.0f} ms**. Target 6 (<= 1 s): **{'met' if med <= 1 else 'not met'}**.\n")


def main():
    REPORT.mkdir(exist_ok=True)
    df = load_manifest()
    out = ["# AgriSense evaluation report\n",
           "_Generated by `scripts/report.py` from the frozen manifest and saved predictions. Do not edit by hand; "
           "hand-written text lives in `report/notes/`._\n"]
    intro = NOTES / "intro.md"
    if intro.exists():
        out.append(intro.read_text(encoding="utf-8"))
    counts = df.groupby(["split", "source"]).size()
    out.append("## 0. Data\n\n| Split | Source | Images |\n|---|---|---|")
    for (s, src), n in counts.items():
        out.append(f"| {s} | {src} | {n} |")
    out.append("")
    main_table(out)
    per_class(out)
    calibration(out)
    risk_coverage(out)
    robustness(out)
    quality_gate(out, df)
    leaf_guard(out, df)
    ablation(out)
    failures(out)
    latency(out)
    disc = NOTES / "discussion.md"
    if disc.exists():
        out.append(disc.read_text(encoding="utf-8"))
    (REPORT / "report.md").write_text("\n".join(out), encoding="utf-8")
    print("wrote", REPORT / "report.md")


if __name__ == "__main__":
    main()
