"""Entraîne YOLO11 et journalise le run dans MLflow.

Paramètres lus dans params.yaml (section train), surchargeables en ligne de
commande pour les expériences :
    python src/models/train.py                       # config de params.yaml
    python src/models/train.py --variant s --epochs 80 --run-name yolo11s

Le serveur MLflow est choisi par la variable MLFLOW_TRACKING_URI
(ex. https://dagshub.com/<compte>/projet-epi.mlflow) ; sans elle, les runs
sont écrits localement dans ./mlruns.
"""
import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

import mlflow
import yaml
from ultralytics import YOLO, settings

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.dataset import class_names, data_yaml, split_stats  # noqa: E402
from src.models.architecture import build_model, count_parameters  # noqa: E402

EXPERIMENT = "epi-detection"
MODELS_DIR = Path("models")


def parse_args() -> dict:
    params = yaml.safe_load(open("params.yaml"))["train"]
    parser = argparse.ArgumentParser()
    for key, value in params.items():
        parser.add_argument(f"--{key.replace('_', '-')}", type=type(value), default=value)
    return vars(parser.parse_args())


def main() -> None:
    p = parse_args()
    # On journalise nous-mêmes : on coupe l'intégration MLflow automatique d'Ultralytics
    settings.update({"mlflow": False})
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name=p["run_name"]) as run:
        model = build_model(p["variant"])
        mlflow.log_params(p)
        mlflow.log_params({"n_parameters": count_parameters(model), "classes": len(class_names())})
        mlflow.log_dict(split_stats(), "dataset/split_stats.json")
        mlflow.set_tags({"dvc_data": "data/processed", "task": "detection"})

        results = model.train(
            data=data_yaml(),
            epochs=p["epochs"],
            imgsz=p["imgsz"],
            batch=p["batch"],
            patience=p["patience"],
            seed=p["seed"],
            mosaic=p["mosaic"],
            mixup=p["mixup"],
            fraction=p["fraction"],
            project="runs",
            name=p["run_name"],
            exist_ok=True,
            deterministic=True,
            plots=True,
        )
        save_dir = Path(results.save_dir)

        # Courbes epoch par epoch (results.csv d'Ultralytics) -> métriques MLflow
        with open(save_dir / "results.csv") as f:
            for row in csv.DictReader(f):
                row = {k.strip(): v for k, v in row.items()}
                step = int(float(row.pop("epoch")))
                mlflow.log_metrics({k.replace("(B)", "").replace(":", "_"): float(v)
                                    for k, v in row.items() if v not in ("", None)}, step=step)

        # Métriques finales sur la validation, avec AP par classe
        best = save_dir / "weights" / "best.pt"
        val = YOLO(str(best)).val(data=data_yaml(), split="val", imgsz=p["imgsz"], plots=False)
        metrics = {"val_mAP50": val.box.map50, "val_mAP50_95": val.box.map,
                   "val_precision": val.box.mp, "val_recall": val.box.mr}
        names = class_names()
        for c in val.box.ap_class_index:
            metrics[f"val_AP50_95_{names[int(c)]}"] = float(val.box.maps[int(c)])
        mlflow.log_metrics(metrics)

        for png in save_dir.glob("*.png"):
            mlflow.log_artifact(str(png), "plots")
        mlflow.log_artifact(str(save_dir / "results.csv"))
        mlflow.log_artifact(str(best), "weights")

        # Sorties suivies par DVC
        MODELS_DIR.mkdir(exist_ok=True)
        shutil.copy2(best, MODELS_DIR / "best.pt")
        (MODELS_DIR / "run_id.txt").write_text(run.info.run_id)
        Path("reports").mkdir(exist_ok=True)
        Path("reports/train.json").write_text(json.dumps({k: round(float(v), 4) for k, v in metrics.items()}, indent=2))
        print(f"Run MLflow {run.info.run_id} : mAP50-95 val = {metrics['val_mAP50_95']:.3f}")


if __name__ == "__main__":
    os.environ.setdefault("MLFLOW_TRACKING_URI", "file:./mlruns")
    main()
