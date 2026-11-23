# Détection des EPI sur chantier (YOLO11 + MLOps)

Projet de fin de module Deep Learning Avancé : industrialisation Vision & MLOps.
Binôme : Anwar Ghourabi et Ghada Lajnef.

Le modèle détecte sur une image de chantier 12 classes : les équipements de protection portés
(`helmet`, `vest`, `gloves`, `goggles`, `boots`), leur absence (`no_helmet`, `no_vest`, `no_gloves`,
`no_goggle`, `no_boots`), ainsi que `Person` et `forklift`. Une API FastAPI sert le modèle exporté en ONNX.

| Élément | Lien |
| --- | --- |
| Code | https://github.com/anwarghourabi/projet-epi |
| Données et modèles (DVC) | https://dagshub.com/ghourabianwar7/projet-epi |
| Suivi MLflow | https://dagshub.com/ghourabianwar7/projet-epi.mlflow |
| Modèle enregistré | `epi-detector` (alias `production`) dans le Model Registry MLflow |

## Résultats actuels

| Run MLflow | Modèle | Image | mAP@0.5 (val) | mAP@[.5:.95] (val) |
| --- | --- | --- | --- | --- |
| baseline-yolo11n | YOLO11n | 640 px | 0,808 | 0,590 |
| yolo11s | YOLO11s | 640 px | 0,838 | 0,622 |
| yolo11s-mixup | YOLO11s + mixup 0,2 | 640 px | 0,835 | 0,622 |
| **yolo11s-800** | **YOLO11s** | **800 px** | **0,865** | **0,646** |

Modèle retenu (yolo11s-800) sur le jeu de **test** : mAP@0.5 = 0,846, mAP@[.5:.95] = 0,629.
Export ONNX FP32 : 0,628 (38 Mo) ; ONNX INT8 : 0,625 (10 Mo, mais plus lent sur CPU).

## Structure

```
data/              raw/ (versionné par DVC), clean/ et processed/ (sorties du pipeline)
src/data/          clean.py (nettoyage), split.py (split 80/10/10 sans fuite, pHash), dataset.py
src/models/        architecture.py, train.py (MLflow), evaluate.py (test + ONNX + INT8), register.py
models/            best.pt, best.onnx, best_int8.onnx (récupérés avec dvc pull, pas dans Git)
reports/           métriques JSON du pipeline (clean, split, train, metrics)
backend/app/       API FastAPI : main.py (routes), inference.py (ONNX Runtime + NMS), schemas.py
backend/tests/     tests de l'API (faux modèle ONNX, pas besoin des données)
backend/Dockerfile image Docker multi-stage de l'API
tests/             test_split.py (aucune image de val/test proche d'une image de train)
notebooks/         train_colab.ipynb (entraînement GPU sur Google Colab)
dvc.yaml           pipeline clean -> split -> train -> evaluate
params.yaml        tous les paramètres (split, entraînement, évaluation)
```

## Installation sur une nouvelle machine (Windows / PowerShell)

Prérequis : **Git**, **Python 3.11 ou plus récent** (`python --version`), et un accès collaborateur
au dépôt GitHub **et** au dépôt DagsHub (à demander au propriétaire : *Settings → Collaborators* sur chacun).

### 1. Cloner le code et se présenter à Git

```powershell
cd D:\projets
git clone https://github.com/anwarghourabi/projet-epi.git
cd projet-epi
git config user.name "Prénom Nom"
git config user.email "email-du-compte-github@exemple.com"
```

Utilisez **votre** nom et l'email de **votre** compte GitHub : c'est ce qui attribue vos commits.

### 2. Créer l'environnement Python

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Si PowerShell refuse d'activer le venv : `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, puis recommencez.
Sur Linux/Mac : `source .venv/bin/activate`.

### 3. Connecter DVC à DagsHub et récupérer les données et le modèle

Sur DagsHub : avatar → *Your Settings* → *Tokens* → copier votre token. Puis :

```powershell
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <votre_utilisateur_dagshub>
dvc remote modify origin --local password <votre_token_dagshub>
dvc pull
```

`--local` écrit les identifiants dans `.dvc/config.local`, qui n'est **jamais** envoyé sur GitHub.
`dvc pull` télécharge environ 650 Mo (images, labels, poids, modèles ONNX). Pour l'API seule,
`dvc pull models/best.onnx` suffit (38 Mo).

### 4. Vérifier que tout fonctionne

```powershell
pytest -q                                   # tests du split et de l'API
uvicorn backend.app.main:app --reload       # lance l'API
```

Ouvrez http://localhost:8000/docs : `GET /health` doit répondre `"model_loaded": true`.
Testez `POST /predict` avec une image de `data\processed\images\test\`.

## Reproduire l'entraînement

### Pipeline de données

```powershell
dvc repro split          # nettoyage puis split 800 / 100 / 100
dvc metrics show
```

### Entraînement suivi dans MLflow (GPU : Google Colab)

Le notebook `notebooks/train_colab.ipynb` enchaîne toutes les étapes sur Colab (secrets `DAGSHUB_USER`,
`DAGSHUB_TOKEN`, `GITHUB_TOKEN`). En ligne de commande, avec un GPU NVIDIA :

```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/ghourabianwar7/projet-epi.mlflow
export MLFLOW_TRACKING_USERNAME=<utilisateur_dagshub>
export MLFLOW_TRACKING_PASSWORD=<token_dagshub>

python src/models/train.py --variant n --run-name baseline-yolo11n --epochs 50
python src/models/train.py --variant s --run-name yolo11s --epochs 50
python src/models/train.py --variant s --run-name yolo11s-mixup --epochs 50 --mixup 0.2
python src/models/train.py --variant s --run-name yolo11s-800 --epochs 50 --imgsz 800 --batch 8
```

Sans ces variables, MLflow écrit en local dans `mlflow.db` (`mlflow ui --backend-store-uri sqlite:///mlflow.db`).

### Version finale : évaluation, export ONNX et Model Registry

La configuration gagnante est dans `params.yaml` (section `train`), puis :

```bash
dvc repro evaluate       # évalue sur le test, exporte ONNX FP32 + INT8, enregistre au Model Registry
dvc push                 # envoie poids et modèles sur DagsHub
```

## API FastAPI

| Route | Méthode | Rôle |
| --- | --- | --- |
| `/health` | GET | état du service, modèle chargé, version, GPU |
| `/model/metadata` | GET | architecture, classes, seuils, métriques de test |
| `/predict` | POST | boîtes, alertes (EPI manquants), conformité, latence serveur |
| `/predict/visualize` | POST | image JPEG annotée |

Erreurs : 400 si le fichier n'est pas une image, 503 si le modèle n'est pas chargé.
Réglages par variables d'environnement : `MODEL_PATH`, `METRICS_PATH`, `CONF_THRESHOLD`, `IOU_THRESHOLD`,
`CONFLICT_IOU`, `MAX_UPLOAD_MB`, `CORS_ORIGINS`.

Exemple : `curl.exe -F "file=@photo.jpg" http://localhost:8000/predict`

### Docker

```powershell
dvc pull models/best.onnx
docker build -f backend/Dockerfile -t epi-api .
docker run --rm -p 8000:8000 epi-api
```

## Prochaines étapes

- Interface web Angular : upload, glisser-déposer, galerie, webcam
- Docker Compose : `docker compose up --build` (API + frontend)
- GitHub Actions : `ci-tests.yml` et `docker-build.yml`
- Amélioration des classes « no_… » (sur-échantillonnage, YOLO11m)
