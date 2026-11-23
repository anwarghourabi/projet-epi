"""API FastAPI du détecteur d'EPI.

Lancement local (depuis la racine du dépôt) :
    uvicorn backend.app.main:app --reload
Documentation interactive : http://localhost:8000/docs
"""
from __future__ import annotations

import io
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image, ImageOps, UnidentifiedImageError

from .inference import ALERT_MESSAGES, Detector, build_alerts, draw_detections
from .schemas import HealthResponse, ModelMetadata, PredictionResponse

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = os.getenv("MODEL_PATH", str(ROOT / "models" / "best.onnx"))
METRICS_PATH = os.getenv("METRICS_PATH", str(ROOT / "reports" / "metrics.json"))
CONF = float(os.getenv("CONF_THRESHOLD", "0.25"))
IOU = float(os.getenv("IOU_THRESHOLD", "0.7"))
CONFLICT_IOU = float(os.getenv("CONFLICT_IOU", "0.45"))  # 1 = désactive la résolution des paires contradictoires
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "10"))
METRIC_KEYS = ["test_mAP50", "test_mAP50_95", "test_precision", "test_recall",
               "onnx_mAP50_95", "latency_ms_onnx_fp32"]

log = logging.getLogger("epi-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle une seule fois au démarrage. S'il est absent, l'API démarre quand même
    et les routes de prédiction répondent 503."""
    try:
        app.state.detector = Detector(MODEL_PATH, conf=CONF, iou=IOU, conflict_iou=CONFLICT_IOU)
        log.info("Modèle chargé : %s (version %s)", MODEL_PATH, app.state.detector.version)
    except Exception as err:  # fichier manquant ou modèle illisible
        app.state.detector = None
        log.error("Modèle non chargé (%s) : %s", MODEL_PATH, err)
    try:
        metrics = json.loads(Path(METRICS_PATH).read_text())
        app.state.metrics = {k: float(metrics[k]) for k in METRIC_KEYS if k in metrics}
    except (OSError, ValueError):
        app.state.metrics = {}
    yield


app = FastAPI(
    title="API de détection des EPI",
    description="Détecte casques, gilets, gants, lunettes, bottes et leur absence sur une image de chantier "
                "(YOLO11s exporté en ONNX).",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Latency-Ms", "X-Alerts", "X-Model-Version"],
)


def get_detector(request: Request) -> Detector:
    detector = request.app.state.detector
    if detector is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé : vérifiez MODEL_PATH.")
    return detector


async def read_image(file: UploadFile) -> Image.Image:
    """Lit l'upload et vérifie que c'est bien une image ; sinon 400."""
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Fichier vide.")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"Image trop lourde (max {MAX_UPLOAD_MB:g} Mo).")
    try:
        Image.open(io.BytesIO(data)).verify()
        image = Image.open(io.BytesIO(data))
        image = ImageOps.exif_transpose(image).convert("RGB")  # respecte l'orientation des photos de téléphone
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        raise HTTPException(status_code=400, detail="Le fichier envoyé n'est pas une image valide (JPEG, PNG…).")
    return image


@app.get("/health", response_model=HealthResponse, tags=["Service"])
def health(request: Request) -> HealthResponse:
    """État du service : modèle chargé ou non, GPU disponible, version du modèle."""
    detector = request.app.state.detector
    return HealthResponse(
        status="ok" if detector else "degraded",
        model_loaded=detector is not None,
        model_version=detector.version if detector else None,
        gpu_available="CUDAExecutionProvider" in ort.get_available_providers(),
        execution_provider=detector.provider if detector else None,
    )


@app.get("/model/metadata", response_model=ModelMetadata, tags=["Modèle"])
def model_metadata(request: Request) -> ModelMetadata:
    """Nom, architecture, classes, seuils et métriques de test du modèle servi."""
    detector = get_detector(request)
    return ModelMetadata(
        name="epi-detector",
        architecture="YOLO11s (Ultralytics), export ONNX FP32",
        version=detector.version,
        input_size=detector.imgsz,
        classes=detector.names,
        alert_classes=list(ALERT_MESSAGES),
        conf_threshold=detector.conf,
        iou_threshold=detector.iou,
        metrics=request.app.state.metrics,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prédiction"])
async def predict(request: Request, file: UploadFile = File(..., description="Image JPEG ou PNG"),
                  conf: float | None = Query(None, ge=0, le=1, description="Seuil de confiance (défaut 0,25)")):
    """Détecte les EPI sur l'image et renvoie les boîtes, les alertes et la latence serveur."""
    detector = get_detector(request)
    image = await read_image(file)
    detections, latency = await run_in_threadpool(detector.predict, image, conf)
    alerts = build_alerts(detections)
    counts: dict[str, int] = {}
    for d in detections:
        counts[d["class_name"]] = counts.get(d["class_name"], 0) + 1
    return PredictionResponse(
        detections=detections, alerts=alerts, compliant=not alerts, counts=counts,
        image_width=image.width, image_height=image.height,
        latency_ms=round(latency, 1), model_version=detector.version,
    )


@app.post("/predict/visualize", tags=["Prédiction"],
          responses={200: {"content": {"image/jpeg": {}}, "description": "Image annotée"}})
async def predict_visualize(request: Request, file: UploadFile = File(..., description="Image JPEG ou PNG"),
                            conf: float | None = Query(None, ge=0, le=1)):
    """Renvoie l'image avec les boîtes dessinées (vert = EPI porté, rouge = EPI manquant)."""
    detector = get_detector(request)
    image = await read_image(file)
    detections, latency = await run_in_threadpool(detector.predict, image, conf)
    annotated = draw_detections(image, detections)
    buf = io.BytesIO()
    annotated.save(buf, format="JPEG", quality=90)
    return Response(content=buf.getvalue(), media_type="image/jpeg", headers={
        "X-Latency-Ms": f"{latency:.1f}",
        "X-Alerts": str(sum(a["count"] for a in build_alerts(detections))),
        "X-Model-Version": detector.version,
    })
