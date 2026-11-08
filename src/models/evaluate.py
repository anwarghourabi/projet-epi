"""Évalue le meilleur modèle sur le jeu de test, l'exporte en ONNX,
le quantifie en INT8, mesure la latence et l'enregistre au Model Registry.

Usage : python src/models/evaluate.py
Le run MLflow complété est celui de models/run_id.txt (écrit par train.py).
"""
import json
import os
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import onnxruntime as ort
import yaml
from mlflow import MlflowClient
from onnxruntime.quantization import QuantType, quantize_dynamic
from ultralytics import YOLO, settings

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.dataset import class_names, data_yaml  # noqa: E402

MODELS = Path("models")
REGISTERED_NAME = "epi-detector"


def test_metrics(weights: Path, imgsz: int, prefix: str) -> dict:
    """mAP sur le split test, pour un .pt ou un .onnx."""
    res = YOLO(str(weights), task="detect").val(data=data_yaml(), split="test", imgsz=imgsz,
                                                batch=1 if weights.suffix == ".onnx" else 16,
                                                plots=weights.suffix == ".pt", project="runs", name="test",
                                                exist_ok=True)
    out = {f"{prefix}_mAP50": res.box.map50, f"{prefix}_mAP50_95": res.box.map,
           f"{prefix}_precision": res.box.mp, f"{prefix}_recall": res.box.mr}
    if weights.suffix == ".pt":
        names = class_names()
        for c in res.box.ap_class_index:
            out[f"test_AP50_95_{names[int(c)]}"] = float(res.box.maps[int(c)])
    return {k: float(v) for k, v in out.items()}, Path(res.save_dir)


def latency_ms(onnx_path: Path, imgsz: int, runs: int = 30) -> float:
    """Latence moyenne d'une inférence ONNX Runtime sur CPU (hors pré/post-traitement)."""
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)
    name = session.get_inputs()[0].name
    for _ in range(5):  # échauffement
        session.run(None, {name: x})
    start = time.perf_counter()
    for _ in range(runs):
        session.run(None, {name: x})
    return (time.perf_counter() - start) / runs * 1000


def register(run_id: str, score: float) -> str:
    """Enregistre le modèle ONNX ; alias 'staging', et 'production' s'il bat la version en production."""
    client = MlflowClient()
    version = mlflow.register_model(f"runs:/{run_id}/onnx", REGISTERED_NAME).version
    client.set_model_version_tag(REGISTERED_NAME, version, "test_mAP50_95", f"{score:.4f}")
    client.set_registered_model_alias(REGISTERED_NAME, "staging", version)
    try:
        prod = client.get_model_version_by_alias(REGISTERED_NAME, "production")
        prod_score = float(prod.tags.get("test_mAP50_95", 0))
    except Exception:
        prod_score = -1.0  # aucune version en production
    if score > prod_score:
        client.set_registered_model_alias(REGISTERED_NAME, "production", version)
        return f"version {version} -> production ({score:.3f} > {prod_score:.3f})"
    return f"version {version} -> staging ({score:.3f} <= production {prod_score:.3f})"


def main() -> None:
    settings.update({"mlflow": False})
    p = yaml.safe_load(open("params.yaml"))
    imgsz = p["train"]["imgsz"]
    run_id = (MODELS / "run_id.txt").read_text().strip()

    metrics, test_dir = test_metrics(MODELS / "best.pt", imgsz, "test")

    # Export ONNX (opset fixe pour la compatibilité avec onnxruntime du conteneur API)
    onnx_path = Path(YOLO(str(MODELS / "best.pt")).export(format="onnx", imgsz=imgsz, opset=17, simplify=True))
    fp32 = MODELS / "best.onnx"
    onnx_path.replace(fp32)
    int8 = MODELS / "best_int8.onnx"
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QUInt8)

    onnx_metrics, _ = test_metrics(fp32, imgsz, "onnx")
    int8_metrics, _ = test_metrics(int8, imgsz, "int8")
    metrics.update(onnx_metrics)
    metrics.update(int8_metrics)
    metrics.update({
        "latency_ms_onnx_fp32": latency_ms(fp32, imgsz),
        "latency_ms_onnx_int8": latency_ms(int8, imgsz),
        "size_mb_onnx_fp32": fp32.stat().st_size / 1e6,
        "size_mb_onnx_int8": int8.stat().st_size / 1e6,
    })

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)
        mlflow.log_params({"conf": p["evaluate"]["conf"], "nms_iou": p["evaluate"]["iou"]})
        for png in test_dir.glob("*.png"):
            mlflow.log_artifact(str(png), "test_plots")
        mlflow.log_artifact(str(fp32), "onnx")
        mlflow.log_artifact(str(int8), "onnx")
        mlflow.log_dict({"names": class_names(), "imgsz": imgsz, **p["evaluate"]}, "onnx/metadata.json")
    try:
        status = register(run_id, metrics["test_mAP50_95"])
    except Exception as err:  # le registre peut être indisponible (serveur local sans base, réseau)
        status = f"non enregistré ({err})"

    Path("reports").mkdir(exist_ok=True)
    Path("reports/metrics.json").write_text(json.dumps({k: round(v, 4) for k, v in metrics.items()}, indent=2))
    print(json.dumps({k: round(v, 4) for k, v in metrics.items() if not k.startswith("test_AP")}, indent=2))
    print("Model Registry :", status)


if __name__ == "__main__":
    os.environ.setdefault("MLFLOW_TRACKING_URI", "file:./mlruns")
    main()
