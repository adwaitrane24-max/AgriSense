"""Train EfficientNet-B0.

    python scripts/train.py --variant baseline   # PlantVillage only
    python scripts/train.py --variant mixed      # PlantVillage + PlantWild (shipping model)

CPU fallbacks from the PRD: --freeze-backbone (head only), --train-blocks N (only the last N
stages plus head), --img-size 160. Keeps the checkpoint with the best tuning-split macro-F1.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agrisense import model as M  # noqa: E402
from agrisense.config import ARTIFACTS, IMG_SIZE, SEED  # noqa: E402
from agrisense.data import ManifestDataset, balanced_sampler, eval_tf, load_manifest, train_tf  # noqa: E402


def evaluate(model, loader):
    model.eval()
    ys, ps = [], []
    with torch.inference_mode():
        for x, y in loader:
            ps.append(model(x).argmax(1).numpy())
            ys.append(y.numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    return f1_score(y, p, average="macro"), (y == p).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["baseline", "mixed"], required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--img-size", type=int, default=IMG_SIZE)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--samples-per-epoch", type=int, default=0, help="0 = size of train split")
    ap.add_argument("--freeze-backbone", action="store_true")
    ap.add_argument("--train-blocks", type=int, default=-1, help="-1 = train everything")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    df = load_manifest()
    if args.variant == "baseline":
        df = df[df.source == "plantvillage"]
    tr, tu = df[df.split == "train"], df[df.split == "tune"]
    n_epoch = args.samples_per_epoch or len(tr)
    print(f"{args.variant}: train={len(tr)} tune={len(tu)} samples/epoch={n_epoch}")

    dl_tr = DataLoader(ManifestDataset(tr, train_tf(args.img_size)), batch_size=args.batch,
                       sampler=balanced_sampler(tr, n_epoch), num_workers=args.workers,
                       persistent_workers=args.workers > 0, drop_last=True)
    dl_tu = DataLoader(ManifestDataset(tu, eval_tf(args.img_size)), batch_size=64,
                       num_workers=args.workers, persistent_workers=args.workers > 0)

    model = M.build()
    head = list(model.get_classifier().parameters())
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    if args.freeze_backbone:
        args.train_blocks = 0
    if args.train_blocks >= 0:
        for p in backbone:
            p.requires_grad = False
        keep = list(model.blocks[len(model.blocks) - args.train_blocks:]) if args.train_blocks else []
        keep += [model.conv_head, model.bn2] if args.train_blocks else []
        for mod in keep:
            for p in mod.parameters():
                p.requires_grad = True
    groups = [{"params": head, "lr": 3e-4}]
    trainable_bb = [p for p in backbone if p.requires_grad]
    if trainable_bb:
        groups.append({"params": trainable_bb, "lr": 1e-4})
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    steps = args.epochs * (n_epoch // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=steps,
                                                pct_start=0.1, anneal_strategy="cos")
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    best, history = -1.0, []
    for ep in range(args.epochs):
        model.train()
        if args.train_blocks >= 0:  # keep frozen BatchNorm stats fixed
            for mod in model.modules():
                if isinstance(mod, torch.nn.BatchNorm2d) and not any(p.requires_grad for p in mod.parameters()):
                    mod.eval()
        t0, tot, n = time.time(), 0.0, 0
        for i, (x, y) in enumerate(dl_tr):
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot, n = tot + loss.item() * len(y), n + len(y)
            if i % 20 == 0:
                print(f"  ep{ep} it{i}/{len(dl_tr)} loss={tot / n:.3f} {time.time() - t0:.0f}s", flush=True)
        f1, acc = evaluate(model, dl_tu)
        history.append({"epoch": ep, "train_loss": tot / n, "tune_macro_f1": f1, "tune_acc": acc,
                        "seconds": time.time() - t0})
        print(f"epoch {ep}: loss={tot / n:.3f} tune macroF1={f1:.3f} acc={acc:.3f} ({time.time() - t0:.0f}s)",
              flush=True)
        if f1 > best:
            best = f1
            M.save(model, args.variant, {"img_size": args.img_size, "epoch": ep, "tune_macro_f1": f1,
                                         "args": vars(args)})
    (ARTIFACTS / f"{args.variant}_train_log.json").write_text(json.dumps(
        {"args": vars(args), "best_tune_macro_f1": best, "history": history}, indent=2))
    print("best tune macro-F1", best)


if __name__ == "__main__":
    main()
