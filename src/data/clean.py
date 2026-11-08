"""Nettoie le dataset brut Construction-PPE.

Fusionne tous les splits présents (on refait le split ensuite), supprime les
labels orphelins, les boîtes dupliquées et les boîtes de moins de 4 px de côté.
Fonctionne avec un dossier data/raw qui contient directement data.yaml ou un
sous-dossier (ex. data/raw/construction-ppe/) qui le contient.
Usage : python src/data/clean.py data/raw data/clean
"""
import json
import shutil
import sys
from pathlib import Path

import yaml
from PIL import Image

MIN_SIDE_PX = 4
IMG_EXT = {".jpg", ".jpeg", ".png"}


def find_root(raw: Path) -> Path:
    """Dossier qui contient data.yaml, images/ et labels/."""
    for yml in sorted(raw.rglob("data.yaml")):
        if (yml.parent / "images").is_dir():
            return yml.parent
    raise FileNotFoundError(f"aucun data.yaml avec un dossier images/ sous {raw}")


def clean(raw: Path, out: Path) -> dict:
    raw = find_root(raw)
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    (out / "labels").mkdir(parents=True)
    report = {"images": 0, "boxes": 0, "orphan_labels": [], "duplicate_boxes": 0,
              "tiny_boxes": 0, "invalid_lines": 0, "images_without_label": [], "corrupt_images": []}

    names = yaml.safe_load((raw / "data.yaml").read_text())["names"]
    report["classes"] = names if isinstance(names, list) else [names[k] for k in sorted(names)]
    (out / "classes.yaml").write_text(yaml.safe_dump({"names": report["classes"]}, allow_unicode=True))

    for split in sorted(p.name for p in (raw / "images").iterdir() if p.is_dir()):
        img_dir, lab_dir = raw / "images" / split, raw / "labels" / split
        stems = {p.stem for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT}
        report["orphan_labels"] += [f"{split}/{p.name}" for p in lab_dir.glob("*.txt") if p.stem not in stems]

        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMG_EXT:
                continue
            try:
                with Image.open(img) as im:
                    im.load()
                    w, h = im.size
            except Exception:
                report["corrupt_images"].append(f"{split}/{img.name}")
                continue
            lab = lab_dir / f"{img.stem}.txt"
            if not lab.exists():
                report["images_without_label"].append(f"{split}/{img.name}")
                continue

            kept, seen = [], set()
            for line in lab.read_text().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    report["invalid_lines"] += 1
                    continue
                c, x, y, bw, bh = int(float(parts[0])), *map(float, parts[1:])
                if not 0 <= c < len(report["classes"]):
                    report["invalid_lines"] += 1
                    continue
                key = (c, round(x, 4), round(y, 4), round(bw, 4), round(bh, 4))
                if key in seen:
                    report["duplicate_boxes"] += 1
                    continue
                if bw * w < MIN_SIDE_PX or bh * h < MIN_SIDE_PX:
                    report["tiny_boxes"] += 1
                    continue
                seen.add(key)
                kept.append(f"{c} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}")

            # Préfixe du split d'origine : évite les collisions de noms entre splits
            name = f"{split}_{img.stem}"
            shutil.copy2(img, out / "images" / f"{name}{img.suffix.lower()}")
            (out / "labels" / f"{name}.txt").write_text("\n".join(kept) + "\n")
            report["images"] += 1
            report["boxes"] += len(kept)
    return report


if __name__ == "__main__":
    raw, out = Path(sys.argv[1]), Path(sys.argv[2])
    rep = clean(raw, out)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/clean.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: (len(v) if isinstance(v, list) and k != "classes" else v) for k, v in rep.items()}, indent=2))
