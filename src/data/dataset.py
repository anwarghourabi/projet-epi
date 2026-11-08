"""Accès au dataset découpé par le pipeline DVC (data/processed)."""
from collections import Counter
from pathlib import Path

import yaml

PROCESSED = Path("data/processed")


def data_yaml(root: Path = PROCESSED) -> str:
    """Chemin absolu du data.yaml, comme l'attend Ultralytics."""
    path = (root / "data.yaml").resolve()
    if not path.exists():
        raise FileNotFoundError(f"{path} introuvable : lancer `dvc pull` ou `dvc repro split`")
    return str(path)


def class_names(root: Path = PROCESSED) -> list[str]:
    names = yaml.safe_load((root / "data.yaml").read_text())["names"]
    return [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)


def split_stats(root: Path = PROCESSED) -> dict:
    """Nombre d'images et de boîtes par classe pour chaque split."""
    names = class_names(root)
    stats = {}
    for split_dir in sorted((root / "labels").iterdir()):
        counts = Counter()
        files = list(split_dir.glob("*.txt"))
        for f in files:
            counts.update(names[int(line.split()[0])] for line in f.read_text().splitlines() if line.strip())
        stats[split_dir.name] = {"images": len(files), "boxes": dict(counts)}
    return stats


if __name__ == "__main__":
    import json

    print(json.dumps(split_stats(), indent=2))
