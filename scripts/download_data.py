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
    """PlantDoc has file names that are invalid on Windows (`?`, `&`), so list the tree with
    git and fetch each tomato/potato file over raw.githubusercontent with a sanitized name."""
    import re
    import urllib.parse
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    meta = RAW / "_plantdoc_meta"
    if not meta.exists():
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--no-checkout",
                        "https://github.com/pratikkayal/PlantDoc-Dataset.git", str(meta)], check=True)
    paths = subprocess.run(["git", "-C", str(meta), "ls-tree", "-r", "--name-only", "HEAD"],
                           check=True, capture_output=True, text=True, encoding="utf-8").stdout.splitlines()
    paths = [p for p in paths if p.split("/")[0] in ("train", "test") and len(p.split("/")) == 3
             and p.split("/")[1].startswith(("Tomato", "Potato"))]
    base = "https://raw.githubusercontent.com/pratikkayal/PlantDoc-Dataset/master/"

    def fetch(p: str) -> None:
        part, folder, name = p.split("/")
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", name)[-120:]
        out = RAW / "plantdoc" / part / folder / safe
        if out.exists():
            return
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(base + urllib.parse.quote(p), timeout=60) as r:
                out.write_bytes(r.read())
        except Exception as e:
            print("fail", p, e)

    with ThreadPoolExecutor(16) as ex:
        list(ex.map(fetch, paths))
    print("plantdoc files:", len(paths))


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

    # Per-file range requests are too slow for ~2,000 files, so fetch the zip once,
    # extract only the admitted folders, then delete the zip to save disk.
    import shutil
    import urllib.request
    import zipfile
    zpath = RAW / "plantwild.zip"
    if not zpath.exists():
        tmp = zpath.with_suffix(".part")
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 22)
        tmp.rename(zpath)
    n = 0
    with zipfile.ZipFile(zpath) as z:
        for m in z.namelist():
            if not m.endswith("/") and keep(m):
                out = RAW / "plantwild" / m
                if not out.exists():
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(z.read(m))
                n += 1
    zpath.unlink()
    print("plantwild: extracted", n)


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
