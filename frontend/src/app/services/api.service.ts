import { HttpClient, HttpErrorResponse, HttpParams } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable, catchError, map, throwError } from 'rxjs';

import { environment } from '../../environments/environment';
import { HealthResponse, ModelMetadata, PredictionResponse, PredictionResult } from '../models';

/** Appels HTTP vers l'API FastAPI. Les erreurs sont traduites en messages lisibles. */
@Injectable({ providedIn: 'root' })
export class ApiService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = environment.apiUrl;

  health(): Observable<HealthResponse> {
    return this.http.get<HealthResponse>(`${this.baseUrl}/health`).pipe(catchError(toMessage));
  }

  metadata(): Observable<ModelMetadata> {
    return this.http.get<ModelMetadata>(`${this.baseUrl}/model/metadata`).pipe(catchError(toMessage));
  }

  /** Envoie l'image à /predict et mesure la latence vue par le navigateur (réseau + serveur). */
  predict(image: Blob, conf: number, filename = 'image.jpg'): Observable<PredictionResult> {
    const form = new FormData();
    form.append('file', image, filename);
    const params = new HttpParams().set('conf', conf);
    const start = performance.now();
    return this.http.post<PredictionResponse>(`${this.baseUrl}/predict`, form, { params }).pipe(
      map((res) => ({ ...res, client_latency_ms: Math.round(performance.now() - start) })),
      catchError(toMessage),
    );
  }
}

function toMessage(err: HttpErrorResponse): Observable<never> {
  let message: string;
  if (err.status === 0) {
    message = "API injoignable : vérifiez qu'elle tourne sur " + environment.apiUrl + ' (uvicorn).';
  } else if (err.status === 400) {
    message = err.error?.detail ?? "Le fichier envoyé n'est pas une image valide.";
  } else if (err.status === 503) {
    message = "Le modèle n'est pas chargé côté API (dvc pull models/best.onnx).";
  } else if (err.status === 422) {
    message = 'Requête invalide : ' + JSON.stringify(err.error?.detail ?? '');
  } else {
    message = `Erreur ${err.status} : ${err.error?.detail ?? err.message}`;
  }
  return throwError(() => new Error(message));
}
