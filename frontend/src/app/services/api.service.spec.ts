import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';

import { environment } from '../../environments/environment';
import { classColor } from '../models';
import { ApiService } from './api.service';

describe('ApiService', () => {
  let api: ApiService;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({ providers: [provideHttpClient(), provideHttpClientTesting()] });
    api = TestBed.inject(ApiService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  it('envoie l’image à /predict avec le seuil et ajoute la latence navigateur', () => {
    let latency = -1;
    api.predict(new Blob(['x'], { type: 'image/jpeg' }), 0.4).subscribe((r) => (latency = r.client_latency_ms));
    const req = http.expectOne(`${environment.apiUrl}/predict?conf=0.4`);
    expect(req.request.method).toBe('POST');
    expect(req.request.body instanceof FormData).toBe(true);
    req.flush({ detections: [], alerts: [], compliant: true, counts: {}, image_width: 10, image_height: 10,
      latency_ms: 5, model_version: 'abc' });
    expect(latency).toBeGreaterThanOrEqual(0);
  });

  it('traduit une API injoignable en message clair', () => {
    let message = '';
    api.health().subscribe({ error: (e: Error) => (message = e.message) });
    http.expectOne(`${environment.apiUrl}/health`).error(new ProgressEvent('error'));
    expect(message).toContain('API injoignable');
  });

  it('reprend le message 400 de l’API', () => {
    let message = '';
    api.predict(new Blob(['x']), 0.25).subscribe({ error: (e: Error) => (message = e.message) });
    http.expectOne(() => true).flush({ detail: 'pas une image' }, { status: 400, statusText: 'Bad Request' });
    expect(message).toBe('pas une image');
  });
});

describe('classColor', () => {
  it('rouge pour un EPI manquant, bleu pour Person, vert pour un EPI porté', () => {
    expect(classColor('no_helmet')).toBe('#dc2626');
    expect(classColor('Person')).toBe('#2563eb');
    expect(classColor('helmet')).toBe('#16a34a');
  });
});
