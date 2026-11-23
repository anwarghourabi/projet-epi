"""Inférence du détecteur d'EPI avec ONNX Runtime (sans PyTorch ni Ultralytics).

Le modèle ONNX exporté par Ultralytics prend une image RGB letterboxée
(1, 3, S, S) normalisée dans [0, 1] et renvoie (1, 4 + nc, N) :
pour chacune des N ancres, la boîte (cx, cy, w, h) en pixels de l'image d'entrée
puis un score par classe (déjà passé par la sigmoïde).
"""
from __future__ import annotations

import ast
import hashlib
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont

# Classes qui signalent un EPI manquant -> alertes de sécurité
ALERT_MESSAGES = {
    "no_helmet": "personne sans casque",
    "no_vest": "personne sans gilet",
    "no_gloves": "mains sans gants",
    "no_boots": "pieds sans chaussures de sécurité",
    "no_goggle": "yeux sans lunettes de protection",
}
DEFAULT_NAMES = ["helmet", "gloves", "vest", "boots", "goggles", "Person",
                 "no_helmet", "no_goggle", "no_gloves", "no_boots", "forklift", "no_vest"]
PAD_COLOR = (114, 114, 114)
# Paires contradictoires : un même objet ne peut pas être à la fois « casque » et « sans casque »
CONFLICT_PAIRS = [("helmet", "no_helmet"), ("vest", "no_vest"), ("gloves", "no_gloves"),
                  ("boots", "no_boots"), ("goggles", "no_goggle")]  # couleur de remplissage utilisée par Ultralytics


def letterbox(image: Image.Image, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Redimensionne en gardant les proportions puis complète en carré size x size.

    Renvoie le tenseur (1, 3, size, size) float32, le facteur d'échelle et le décalage (gauche, haut).
    """
    w, h = image.size
    r = min(size / w, size / h)
    new_w, new_h = round(w * r), round(h * r)
    left = round((size - new_w) / 2 - 0.1)
    top = round((size - new_h) / 2 - 0.1)
    canvas = Image.new("RGB", (size, size), PAD_COLOR)
    canvas.paste(image.convert("RGB").resize((new_w, new_h), Image.BILINEAR), (left, top))
    x = np.asarray(canvas, dtype=np.float32) / 255.0
    return x.transpose(2, 0, 1)[None], r, (left, top)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Non-Maximum Suppression : garde la meilleure boîte et supprime celles qui la recouvrent trop."""
    order = scores.argsort()[::-1]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_threshold]
    return keep


class Detector:
    """Charge le modèle ONNX une seule fois et fait prétraitement, inférence et post-traitement."""

    def __init__(self, model_path: str | Path, conf: float = 0.25, iou: float = 0.7, max_det: int = 300,
                 conflict_iou: float = 0.45):
        self.model_path = Path(model_path)
        self.conf, self.iou, self.max_det, self.conflict_iou = conf, iou, max_det, conflict_iou
        # GPU NVIDIA si onnxruntime-gpu est installé, sinon CPU
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        meta = self.session.get_modelmeta().custom_metadata_map  # métadonnées écrites par Ultralytics
        self.names = self._read_names(meta)
        imgsz = ast.literal_eval(meta["imgsz"]) if "imgsz" in meta else self.session.get_inputs()[0].shape[2:]
        self.imgsz = int(imgsz[0])
        self.version = hashlib.sha256(self.model_path.read_bytes()).hexdigest()[:12]
        self.provider = self.session.get_providers()[0]

    @staticmethod
    def _read_names(meta: dict) -> list[str]:
        if "names" not in meta:
            return DEFAULT_NAMES
        names = ast.literal_eval(meta["names"])  # ex. "{0: 'helmet', 1: 'gloves', ...}"
        return [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)

    def predict(self, image: Image.Image, conf: float | None = None) -> tuple[list[dict], float]:
        """Renvoie les détections (coordonnées en pixels de l'image d'origine) et la latence en ms."""
        conf = self.conf if conf is None else conf
        start = time.perf_counter()
        x, r, (left, top) = letterbox(image, self.imgsz)
        out = self.session.run(None, {self.input_name: x})[0][0].T  # (N, 4 + nc)
        scores_all = out[:, 4:]
        cls = scores_all.argmax(1)
        scores = scores_all[np.arange(len(cls)), cls]
        mask = scores >= conf
        boxes_xywh, scores, cls = out[mask, :4], scores[mask], cls[mask]

        boxes = np.empty_like(boxes_xywh)
        boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
        boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
        # NMS par classe : on décale les boîtes de chaque classe pour qu'elles ne se suppriment pas entre elles
        keep = nms(boxes + cls[:, None] * 7680.0, scores, self.iou)[: self.max_det]

        w, h = image.size
        detections = []
        for i in keep:
            x1, y1, x2, y2 = boxes[i]
            x1, x2 = np.clip([(x1 - left) / r, (x2 - left) / r], 0, w)
            y1, y2 = np.clip([(y1 - top) / r, (y2 - top) / r], 0, h)
            detections.append({
                "class_id": int(cls[i]),
                "class_name": self.names[int(cls[i])],
                "confidence": round(float(scores[i]), 4),
                "bbox": {"x1": round(float(x1), 1), "y1": round(float(y1), 1),
                         "x2": round(float(x2), 1), "y2": round(float(y2), 1)},
            })
        detections = resolve_conflicts(detections, self.conflict_iou)
        return detections, (time.perf_counter() - start) * 1000


def box_iou(a: dict, b: dict) -> float:
    """IoU entre deux boîtes {x1, y1, x2, y2}."""
    iw = max(0.0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]))
    ih = max(0.0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))
    inter = iw * ih
    union = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"]) + (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]) - inter
    return inter / union if union > 0 else 0.0


def resolve_conflicts(detections: list[dict], iou_threshold: float = 0.45) -> list[dict]:
    """Si « helmet » et « no_helmet » (ou vest/no_vest…) se recouvrent au même endroit,
    le modèle hésite sur le même objet : on garde seulement la classe la plus confiante."""
    if iou_threshold >= 1:
        return detections
    dropped = set()
    for pos, neg in CONFLICT_PAIRS:
        for i, a in enumerate(detections):
            if a["class_name"] != neg:
                continue
            for j, b in enumerate(detections):
                if b["class_name"] == pos and box_iou(a["bbox"], b["bbox"]) > iou_threshold:
                    dropped.add(i if a["confidence"] < b["confidence"] else j)
    return [d for k, d in enumerate(detections) if k not in dropped]


def build_alerts(detections: list[dict]) -> list[dict]:
    """Une alerte par classe « no_… » détectée, avec le nombre d'occurrences."""
    counts: dict[str, int] = {}
    for d in detections:
        if d["class_name"] in ALERT_MESSAGES:
            counts[d["class_name"]] = counts.get(d["class_name"], 0) + 1
    return [{"class_name": name, "count": n, "message": f"{n} {ALERT_MESSAGES[name]}"}
            for name, n in sorted(counts.items(), key=lambda kv: -kv[1])]


def draw_detections(image: Image.Image, detections: list[dict]) -> Image.Image:
    """Dessine les boîtes : vert = EPI porté, rouge = EPI manquant, bleu = autres classes."""
    img = image.convert("RGB")
    draw = ImageDraw.Draw(img)
    thickness = max(2, round(max(img.size) / 400))
    font = ImageFont.load_default(size=max(12, round(max(img.size) / 60)))
    for d in detections:
        name = d["class_name"]
        color = (220, 38, 38) if name in ALERT_MESSAGES else (37, 99, 235) if name in ("Person", "forklift") else (22, 163, 74)
        b = d["bbox"]
        draw.rectangle([b["x1"], b["y1"], b["x2"], b["y2"]], outline=color, width=thickness)
        label = f"{name} {d['confidence']:.2f}"
        tx1, ty1, tx2, ty2 = draw.textbbox((b["x1"], b["y1"]), label, font=font)
        th = ty2 - ty1 + 4
        y = b["y1"] - th if b["y1"] - th >= 0 else b["y1"]
        draw.rectangle([b["x1"], y, b["x1"] + (tx2 - tx1) + 6, y + th], fill=color)
        draw.text((b["x1"] + 3, y + 1), label, fill="white", font=font)
    return img
