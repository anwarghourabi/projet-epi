"""Fixtures : un faux modèle ONNX au format YOLO11 pour tester l'API sans le vrai modèle (ni GPU, ni DVC)."""
import io
import importlib

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from PIL import Image

NAMES = ["helmet", "gloves", "vest", "boots", "goggles", "Person",
         "no_helmet", "no_goggle", "no_gloves", "no_boots", "forklift", "no_vest"]
IMGSZ = 640


def fake_output() -> np.ndarray:
    """Sortie fixe (1, 16, 4) : 4 ancres, boîtes (cx, cy, w, h) en pixels de l'entrée 640 x 640."""
    out = np.zeros((1, 4 + len(NAMES), 4), dtype=np.float32)
    out[0, :4, 0] = [320, 320, 100, 100]; out[0, 4 + 0, 0] = 0.90   # helmet, gardé
    out[0, :4, 1] = [325, 322, 100, 100]; out[0, 4 + 0, 1] = 0.80   # doublon du précédent -> supprimé par la NMS
    out[0, :4, 2] = [100, 300, 50, 60];   out[0, 4 + 6, 2] = 0.60   # no_helmet -> alerte
    out[0, :4, 3] = [500, 300, 40, 40];   out[0, 4 + 2, 3] = 0.10   # vest sous le seuil 0,25 -> ignoré
    return out


@pytest.fixture(scope="session")
def fake_model(tmp_path_factory):
    path = tmp_path_factory.mktemp("model") / "best.onnx"
    images = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, IMGSZ, IMGSZ])
    output = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 16, 4])
    nodes = [
        helper.make_node("ReduceMean", ["images"], ["m"], keepdims=0),
        helper.make_node("Mul", ["m", "zero"], ["z"]),
        helper.make_node("Add", ["const", "z"], ["output0"]),  # sortie fixe, qui dépend quand même de l'entrée
    ]
    inits = [numpy_helper.from_array(fake_output(), "const"),
             numpy_helper.from_array(np.array(0, dtype=np.float32), "zero")]
    graph = helper.make_graph(nodes, "fake_yolo", [images], [output], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    for k, v in {"names": str(dict(enumerate(NAMES))), "imgsz": f"[{IMGSZ}, {IMGSZ}]"}.items():
        model.metadata_props.add(key=k, value=v)
    onnx.save(model, path)
    return path


def make_client(monkeypatch, model_path, metrics_path=None):
    from fastapi.testclient import TestClient
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    if metrics_path:
        monkeypatch.setenv("METRICS_PATH", str(metrics_path))
    import backend.app.main as main
    importlib.reload(main)  # relit les variables d'environnement
    return TestClient(main.app)


@pytest.fixture
def client(monkeypatch, fake_model, tmp_path):
    metrics = tmp_path / "metrics.json"
    metrics.write_text('{"test_mAP50": 0.8461, "test_mAP50_95": 0.6291, "other": 1}')
    with make_client(monkeypatch, fake_model, metrics) as c:
        yield c


@pytest.fixture
def client_without_model(monkeypatch, tmp_path):
    with make_client(monkeypatch, tmp_path / "absent.onnx") as c:
        yield c


def image_bytes(width=1280, height=640, fmt="JPEG") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (200, 180, 150)).save(buf, format=fmt)
    return buf.getvalue()
