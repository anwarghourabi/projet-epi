"""Refait le découpage train/val/test sans fuite.

Les images quasi identiques (pHash, distance de Hamming <= seuil) sont
regroupées, et chaque groupe entier va dans un seul split.
Usage : python src/data/split.py data/clean data/processed
"""
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import imagehash
import yaml
from PIL import Image

def find(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def group_near_duplicates(images, threshold):
    hashes = [imagehash.phash(Image.open(p)) for p in images]
    parent = list(range(len(images)))
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            if hashes[i] - hashes[j] <= threshold:
                parent[find(parent, i)] = find(parent, j)
    groups = {}
    for i in range(len(images)):
        groups.setdefault(find(parent, i), []).append(images[i])
    return list(groups.values())


def split(clean: Path, out: Path, params: dict) -> dict:
    names = yaml.safe_load((clean / "classes.yaml").read_text())["names"]
    images = sorted((clean / "images").iterdir())
    groups = group_near_duplicates(images, params["phash_threshold"])
    random.Random(params["seed"]).shuffle(groups)

    n = len(images)
    targets = {"val": params["val_ratio"] * n, "test": params["test_ratio"] * n}
    assign = {"train": [], "val": [], "test": []}
    for g in groups:
        # Remplit val puis test jusqu'à leur quota, le reste va dans train
        for s in ("val", "test"):
            if sum(len(x) for x in assign[s]) + len(g) <= targets[s]:
                assign[s].append(g)
                break
        else:
            assign["train"].append(g)

    if out.exists():
        shutil.rmtree(out)
    report = {"groups": len(groups), "largest_group": max(len(g) for g in groups), "splits": {}}
    for s, gs in assign.items():
        (out / "images" / s).mkdir(parents=True)
        (out / "labels" / s).mkdir(parents=True)
        counts = Counter()
        for g in gs:
            for img in g:
                shutil.copy2(img, out / "images" / s / img.name)
                lab = clean / "labels" / f"{img.stem}.txt"
                shutil.copy2(lab, out / "labels" / s / lab.name)
                counts.update(names[int(l.split()[0])] for l in lab.read_text().splitlines() if l.strip())
        report["splits"][s] = {"images": sum(len(g) for g in gs), "boxes": dict(counts)}

    (out / "data.yaml").write_text(yaml.safe_dump({
        # Pas de clé "path" : le chemin absolu changerait d'un ordinateur à l'autre.
        # Donner le chemin de ce fichier à YOLO suffit.
        "train": "images/train", "val": "images/val",
        "test": "images/test", "names": dict(enumerate(names))}, sort_keys=False))
    return report


if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml"))["split"]
    rep = split(Path(sys.argv[1]), Path(sys.argv[2]), params)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/split.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps({s: v["images"] for s, v in rep["splits"].items()}, indent=2))
