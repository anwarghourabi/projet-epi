# Image de l'API de détection des EPI — build multi-stage.
# À lancer depuis la racine du dépôt (le modèle est dans models/) :
#   dvc pull models/best.onnx
#   docker build -f backend/Dockerfile -t epi-api .
#   docker run -p 8000:8000 epi-api

# --- Étape 1 : installation des dépendances (outils de build, cache pip) ---
FROM python:3.12-slim AS builder
WORKDIR /build
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- Étape 2 : image finale légère, sans outils de build ---
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/best.onnx \
    METRICS_PATH=/app/reports/metrics.json
WORKDIR /app
COPY --from=builder /install /usr/local
COPY backend/app ./backend/app
COPY models/best.onnx ./models/best.onnx
COPY reports/metrics.json ./reports/metrics.json
RUN useradd --create-home api && chown -R api /app
USER api
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
