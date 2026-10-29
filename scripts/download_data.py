"""Download only the tomato/potato parts of PlantVillage, PlantDoc and PlantWild,
plus ~120 COCO images for the non-leaf set. Nothing here goes into git.

    python scripts/download_data.py            # everything
    python scripts/download_data.py plantwild  # one source
"""
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agrisense.config import AUX, PLANTVILLAGE_MAP, PLANTWILD_MAP, RAW, SEED  # noqa: E402


def sparse_clone(url: str, dest: Path, patterns: list[str]) -> None:
    if dest.exists():
        print(f"[skip] {dest} exists")
        return
    subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--no-checkout",
                    url, str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "sparse-checkout", "init", "--no-cone"], check=True)
    subprocess.run(["git", "-C", str(dest), "sparse-checkout", "set", "--no-cone", *patterns], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout"], check=True)


def plantvillage() -> None:
    sparse_clone("https://github.com/spMohanty/PlantVillage-Dataset.git", RAW / "plantvillage",
                 [f"/raw/color/{d}/" for d in PLANTVILLAGE_MAP])


def plantdoc() -> None:
    sparse_clone("https://github.com/pratikkayal/PlantDoc-Dataset.git", RAW / "plantdoc",
                 ["/train/Tomato*", "/train/Potato*", "/test/Tomato*", "/test/Potato*"])


def remote_extract(url: str, dest: Path, keep) -> int:
    from remotezip import RemoteZip
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    with RemoteZip(url) as z:
        members = [m for m in z.namelist() if not m.endswith("/") and keep(m)]
        for i, m in enumerate(members):
            out = dest / m
            if out.exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(z.read(m))
            n += 1
            if i % 200 == 0:
                print(f"  {i}/{len(members)}")
    return n


def plantwild() -> None:
    url = "https://huggingface.co/datasets/uqtwei2/PlantWild/resolve/main/plantwild.zip"

    def keep(m: str) -> bool:
        parts = m.split("/")
        return len(parts) == 4 and parts[2] in PLANTWILD_MAP and \
            parts[3].lower().endswith((".jpg", ".jpeg", ".png", ".webp"))

    print("plantwild: extracted", remote_extract(url, RAW / "plantwild", keep))


def nonleaf(n: int = 120) -> None:
    """Random COCO val2017 photos: people, hands, food, fruit, objects, street scenes."""
    from remotezip import RemoteZip
    dest = AUX / "nonleaf"
    if dest.exists() and len(list(dest.glob("*.jpg"))) >= n:
        print("[skip] nonleaf exists")
        return
    dest.mkdir(parents=True, exist_ok=True)
    with RemoteZip("http://images.cocodataset.org/zips/val2017.zip") as z:
        names = sorted(m for m in z.namelist() if m.endswith(".jpg"))
        random.Random(SEED).shuffle(names)
        for m in names[:n]:
            (dest / Path(m).name).write_bytes(z.read(m))
    print("nonleaf:", n)


SOURCES = {"plantvillage": plantvillage, "plantdoc": plantdoc, "plantwild": plantwild, "nonleaf": nonleaf}

if __name__ == "__main__":
    for name in sys.argv[1:] or SOURCES:
        print(f"== {name}")
        SOURCES[name]()
