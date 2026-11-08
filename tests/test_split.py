"""Vérifie qu'aucune image de val/test n'est un quasi-doublon d'une image de train."""
from pathlib import Path

import imagehash
import pytest
from PIL import Image

ROOT = Path("data/processed/images")


@pytest.mark.skipif(not ROOT.exists(), reason="lancer `dvc repro` d'abord")
def test_no_leak_between_splits():
    train = [imagehash.phash(Image.open(p)) for p in (ROOT / "train").iterdir()]
    for split in ("val", "test"):
        for p in (ROOT / split).iterdir():
            h = imagehash.phash(Image.open(p))
            assert all(h - t > 4 for t in train), f"{split}/{p.name} fuit vers train"
