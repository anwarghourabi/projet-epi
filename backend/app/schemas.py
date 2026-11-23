"""Schémas Pydantic des réponses de l'API (documentés automatiquement dans /docs)."""
from pydantic import BaseModel, ConfigDict, Field


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BBox(Strict):
    x1: float = Field(..., description="Bord gauche, en pixels de l'image envoyée")
    y1: float = Field(..., description="Bord haut")
    x2: float = Field(..., description="Bord droit")
    y2: float = Field(..., description="Bord bas")


class Box(Strict):
    class_id: int = Field(..., ge=0)
    class_name: str = Field(..., examples=["helmet"])
    confidence: float = Field(..., ge=0, le=1)
    bbox: BBox


class Alert(Strict):
    class_name: str = Field(..., examples=["no_helmet"])
    count: int = Field(..., ge=1)
    message: str = Field(..., examples=["2 personne sans casque"])


class PredictionResponse(Strict):
    detections: list[Box]
    alerts: list[Alert] = Field(..., description="Une alerte par type d'EPI manquant détecté")
    compliant: bool = Field(..., description="Vrai si aucun EPI manquant n'est détecté")
    counts: dict[str, int] = Field(..., description="Nombre de détections par classe")
    image_width: int
    image_height: int
    latency_ms: float = Field(..., description="Temps d'inférence côté serveur (prétraitement + ONNX + NMS)")
    model_version: str


class HealthResponse(Strict):
    status: str = Field(..., examples=["ok"])
    model_loaded: bool
    model_version: str | None
    gpu_available: bool
    execution_provider: str | None


class ModelMetadata(Strict):
    name: str
    architecture: str
    version: str
    input_size: int
    classes: list[str]
    alert_classes: list[str]
    conf_threshold: float
    iou_threshold: float
    metrics: dict[str, float] = Field(..., description="Métriques de test issues de reports/metrics.json")
