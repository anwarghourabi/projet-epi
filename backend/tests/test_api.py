"""Tests de l'API : chaque route, image invalide, modèle absent et fonctions d'inférence."""
import io

import numpy as np
import pytest
from PIL import Image

from backend.app.inference import build_alerts, letterbox, nms, resolve_conflicts
from conftest import NAMES, image_bytes


# ---------- Routes ----------

def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert len(body["model_version"]) == 12
    assert isinstance(body["gpu_available"], bool)


def test_model_metadata(client):
    r = client.get("/model/metadata")
    assert r.status_code == 200
    body = r.json()
    assert body["classes"] == NAMES
    assert body["input_size"] == 640
    assert "no_helmet" in body["alert_classes"]
    assert body["metrics"] == {"test_mAP50": 0.8461, "test_mAP50_95": 0.6291}  # clés inconnues filtrées


def test_predict_boxes_nms_and_alerts(client):
    r = client.post("/predict", files={"file": ("chantier.jpg", image_bytes(1280, 640), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    names = [d["class_name"] for d in body["detections"]]
    assert sorted(names) == ["helmet", "no_helmet"]          # doublon supprimé, score 0,10 filtré
    helmet = next(d for d in body["detections"] if d["class_name"] == "helmet")
    # boîte 640 px -> image 1280 x 640 : échelle 0,5 et bande grise de 160 px en haut
    assert helmet["bbox"] == {"x1": 540.0, "y1": 220.0, "x2": 740.0, "y2": 420.0}
    assert helmet["confidence"] == pytest.approx(0.9)
    assert body["alerts"] == [{"class_name": "no_helmet", "count": 1, "message": "1 personne sans casque"}]
    assert body["compliant"] is False
    assert body["counts"] == {"helmet": 1, "no_helmet": 1}
    assert (body["image_width"], body["image_height"]) == (1280, 640)
    assert body["latency_ms"] >= 0


def test_predict_conf_threshold(client):
    r = client.post("/predict?conf=0.7", files={"file": ("a.png", image_bytes(640, 640, "PNG"), "image/png")})
    assert r.status_code == 200
    assert [d["class_name"] for d in r.json()["detections"]] == ["helmet"]
    assert r.json()["compliant"] is True


def test_predict_visualize_returns_jpeg(client):
    r = client.post("/predict/visualize", files={"file": ("a.jpg", image_bytes(800, 600), "image/jpeg")})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert r.headers["x-alerts"] == "1"
    assert float(r.headers["x-latency-ms"]) >= 0
    assert Image.open(io.BytesIO(r.content)).size == (800, 600)


# ---------- Erreurs ----------

@pytest.mark.parametrize("route", ["/predict", "/predict/visualize"])
def test_invalid_image_returns_400(client, route):
    r = client.post(route, files={"file": ("notes.txt", b"ceci n'est pas une image", "text/plain")})
    assert r.status_code == 400


def test_empty_file_returns_400(client):
    r = client.post("/predict", files={"file": ("vide.jpg", b"", "image/jpeg")})
    assert r.status_code == 400


def test_missing_file_returns_422(client):
    assert client.post("/predict").status_code == 422


def test_model_absent_returns_503(client_without_model):
    c = client_without_model
    assert c.get("/health").json() == {"status": "degraded", "model_loaded": False, "model_version": None,
                                      "gpu_available": c.get("/health").json()["gpu_available"],
                                      "execution_provider": None}
    assert c.get("/model/metadata").status_code == 503
    r = c.post("/predict", files={"file": ("a.jpg", image_bytes(), "image/jpeg")})
    assert r.status_code == 503


# ---------- Fonctions d'inférence ----------

def test_letterbox_shape_and_padding():
    x, r, (left, top) = letterbox(Image.new("RGB", (1280, 640), "white"), 640)
    assert x.shape == (1, 3, 640, 640) and x.dtype == np.float32
    assert r == 0.5 and (left, top) == (0, 160)
    assert x.max() <= 1.0
    assert np.allclose(x[0, :, 0, 0], 114 / 255)   # bande de remplissage grise
    assert np.allclose(x[0, :, 320, 320], 1.0)     # image blanche au centre


def test_nms_keeps_best_and_separate_boxes():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    assert nms(boxes, scores, 0.5) == [0, 2]


def test_build_alerts_counts_missing_ppe():
    dets = [{"class_name": n} for n in ["no_vest", "helmet", "no_vest", "no_gloves"]]
    alerts = build_alerts(dets)
    assert alerts[0] == {"class_name": "no_vest", "count": 2, "message": "2 personne sans gilet"}
    assert {a["class_name"] for a in alerts} == {"no_vest", "no_gloves"}


def test_conflicting_pair_keeps_most_confident():
    def det(name, conf, x1):
        return {"class_name": name, "confidence": conf, "bbox": {"x1": x1, "y1": 0, "x2": x1 + 100, "y2": 100}}
    dets = [det("helmet", 0.85, 0), det("no_helmet", 0.30, 5),   # même tête : on garde helmet
            det("no_vest", 0.73, 300), det("vest", 0.20, 400)]   # boîtes éloignées : on garde les deux
    names = [d["class_name"] for d in resolve_conflicts(dets, 0.45)]
    assert names == ["helmet", "no_vest", "vest"]
    assert len(resolve_conflicts(dets, 1.0)) == 4                # 1 = désactivé
