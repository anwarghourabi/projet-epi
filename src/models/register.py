"""Enregistre models/best.onnx au Model Registry MLflow (modèle "epi-detector")."""
import json
import os
from pathlib import Path

import mlflow
import onnx
from mlflow import MlflowClient

MODELS = Path("models")
REGISTERED_NAME = "epi-detector"


def register(run_id: str, score: float, onnx_path: Path = MODELS / "best.onnx") -> str:
    with mlflow.start_run(run_id=run_id):
        info = mlflow.onnx.log_model(onnx.load(str(onnx_path)), name="onnx-model")
    version = mlflow.register_model(info.model_uri, REGISTERED_NAME).version
    client = MlflowClient()
    client.set_model_version_tag(REGISTERED_NAME, version, "test_mAP50_95", f"{score:.4f}")
    client.set_registered_model_alias(REGISTERED_NAME, "staging", version)
    try:
        prod = client.get_model_version_by_alias(REGISTERED_NAME, "production")
        prod_score = float(prod.tags.get("test_mAP50_95", 0))
    except Exception:
        prod_score = -1.0
    if score > prod_score:
        client.set_registered_model_alias(REGISTERED_NAME, "production", version)
        return f"version {version} -> production ({score:.3f} > {prod_score:.3f})"
    return f"version {version} -> staging ({score:.3f} <= production {prod_score:.3f})"


if __name__ == "__main__":
    os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    run_id = (MODELS / "run_id.txt").read_text().strip()
    score = json.loads(Path("reports/metrics.json").read_text())["test_mAP50_95"]
    print("Model Registry :", register(run_id, score))
