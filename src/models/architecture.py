"""Choix du détecteur YOLO11.

Les poids pré-entraînés sur COCO sont téléchargés automatiquement par
Ultralytics au premier appel ; la tête de détection est réadaptée aux
12 classes du dataset au début de l'entraînement (transfer learning).
"""
from ultralytics import YOLO

# Taille -> poids pré-entraînés COCO. n = baseline, s et m = modèles avancés.
VARIANTS = {
    "n": "yolo11n.pt",  # 2,6 M paramètres
    "s": "yolo11s.pt",  # 9,4 M paramètres
    "m": "yolo11m.pt",  # 20,1 M paramètres
}


def build_model(variant: str = "n") -> YOLO:
    if variant not in VARIANTS:
        raise ValueError(f"variante inconnue {variant!r}, choisir parmi {list(VARIANTS)}")
    return YOLO(VARIANTS[variant])


def count_parameters(model: YOLO) -> int:
    return sum(p.numel() for p in model.model.parameters())
