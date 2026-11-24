/** Types des réponses de l'API (miroir de backend/app/schemas.py). */
export interface BBox { x1: number; y1: number; x2: number; y2: number; }

export interface Detection {
  class_id: number;
  class_name: string;
  confidence: number;
  bbox: BBox;
}

export interface Alert { class_name: string; count: number; message: string; }

export interface PredictionResponse {
  detections: Detection[];
  alerts: Alert[];
  compliant: boolean;
  counts: Record<string, number>;
  image_width: number;
  image_height: number;
  latency_ms: number;
  model_version: string;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  model_version: string | null;
  gpu_available: boolean;
  execution_provider: string | null;
}

export interface ModelMetadata {
  name: string;
  architecture: string;
  version: string;
  input_size: number;
  classes: string[];
  alert_classes: string[];
  conf_threshold: number;
  iou_threshold: number;
  metrics: Record<string, number>;
}

/** Résultat affiché : la réponse de l'API + la latence mesurée côté navigateur. */
export interface PredictionResult extends PredictionResponse { client_latency_ms: number; }

/** Couleur d'une classe : rouge = EPI manquant, bleu = personne / chariot, vert = EPI porté. */
export function classColor(name: string): string {
  if (name.startsWith('no_')) return '#dc2626';
  if (name === 'Person' || name === 'forklift') return '#2563eb';
  return '#16a34a';
}
