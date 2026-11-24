# Interface web (Angular)

Interface qui appelle l'API FastAPI du détecteur d'EPI : dépôt d'image (glisser-déposer ou clic),
galerie d'exemples, webcam, boîtes dessinées sur l'image, alertes, latences serveur et navigateur.

## Lancer en local

Prérequis : Node.js 22 LTS (`node --version`).

```powershell
# Terminal 1, à la racine du dépôt : l'API
uvicorn backend.app.main:app --reload

# Terminal 2 : l'interface
cd frontend
npm install
npm start            # http://localhost:4200
```

L'adresse de l'API se règle dans `src/environments/environment.ts` (par défaut `http://localhost:8000`).

## Commandes

| Commande | Rôle |
| --- | --- |
| `npm start` | serveur de développement (rechargement automatique) |
| `npm run build` | build de production dans `dist/frontend/browser` |
| `npm test -- --watch=false` | tests unitaires (Vitest) |
| `npm run lint` | ESLint (angular-eslint) |

## Organisation

```
src/app/
  services/api.service.ts      appels /health, /model/metadata, /predict + messages d'erreur
  models.ts                    types des réponses (miroir de backend/app/schemas.py), couleurs
  components/
    upload-zone/               glisser-déposer + choix de fichier (refuse les non-images)
    example-gallery/           images de public/examples/ (liste dans examples.json)
    webcam/                    getUserMedia, une image envoyée dès que la précédente est traitée
    detection-viewer/          canvas : image + boîtes (vert porté, rouge manquant, bleu personne)
    results-panel/             conformité, alertes, latences, barres de score par détection
  app.ts / app.html            page : onglets Image / Webcam, seuil de confiance, état de l'API
```

Pour ajouter des exemples : copier des images de `data/processed/images/test/` dans `public/examples/`
et les déclarer dans `public/examples/examples.json`.
