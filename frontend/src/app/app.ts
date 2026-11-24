import { DecimalPipe } from '@angular/common';
import { Component, OnDestroy, inject, signal } from '@angular/core';

import { DetectionViewer } from './components/detection-viewer/detection-viewer';
import { ExampleGallery } from './components/example-gallery/example-gallery';
import { ResultsPanel } from './components/results-panel/results-panel';
import { UploadZone } from './components/upload-zone/upload-zone';
import { Webcam } from './components/webcam/webcam';
import { HealthResponse, ModelMetadata, PredictionResult } from './models';
import { ApiService } from './services/api.service';

type Mode = 'image' | 'webcam';

@Component({
  selector: 'app-root',
  imports: [DecimalPipe, DetectionViewer, ExampleGallery, ResultsPanel, UploadZone, Webcam],
  templateUrl: './app.html',
})
export class App implements OnDestroy {
  private readonly api = inject(ApiService);

  protected readonly mode = signal<Mode>('image');
  protected readonly conf = signal(0.25);
  protected readonly imageSrc = signal<string | null>(null);
  protected readonly result = signal<PredictionResult | null>(null);
  protected readonly loading = signal(false);
  protected readonly error = signal<string | null>(null);
  protected readonly health = signal<HealthResponse | null>(null);
  protected readonly apiDown = signal(false);
  protected readonly metadata = signal<ModelMetadata | null>(null);

  private lastFile: File | null = null;

  constructor() {
    this.checkApi();
  }

  protected checkApi(): void {
    this.api.health().subscribe({
      next: (h) => {
        this.health.set(h);
        this.apiDown.set(false);
        if (h.model_loaded) this.api.metadata().subscribe({ next: (m) => this.metadata.set(m), error: () => undefined });
      },
      error: (e: Error) => {
        this.apiDown.set(true);
        this.error.set(e.message);
      },
    });
  }

  protected setMode(mode: Mode): void {
    this.mode.set(mode);
    this.reset();
  }

  /** Image choisie (dépôt, clic ou galerie) : on l'affiche tout de suite puis on lance la détection. */
  protected analyzeFile(file: File): void {
    this.lastFile = file;
    this.showImage(file);
    this.result.set(null);
    this.send(file, file.name);
  }

  /** Image capturée par la webcam : on n'affiche l'image qu'avec ses boîtes, pour qu'elles correspondent. */
  protected analyzeFrame(frame: Blob): void {
    this.send(frame, 'webcam.jpg', true);
  }

  protected onConfChange(event: Event): void {
    this.conf.set(Number((event.target as HTMLInputElement).value));
    if (this.mode() === 'image' && this.lastFile) this.send(this.lastFile, this.lastFile.name);
  }

  protected showError(message: string): void {
    this.error.set(message);
  }

  private send(image: Blob, name: string, isFrame = false): void {
    this.loading.set(true);
    this.error.set(null);
    this.api.predict(image, this.conf(), name).subscribe({
      next: (res) => {
        if (isFrame) this.showImage(image);
        this.result.set(res);
        this.loading.set(false);
      },
      error: (e: Error) => {
        this.error.set(e.message);
        this.loading.set(false);
      },
    });
  }

  private showImage(image: Blob): void {
    const old = this.imageSrc();
    if (old) URL.revokeObjectURL(old);
    this.imageSrc.set(URL.createObjectURL(image));
  }

  private reset(): void {
    const old = this.imageSrc();
    if (old) URL.revokeObjectURL(old);
    this.imageSrc.set(null);
    this.result.set(null);
    this.error.set(null);
    this.lastFile = null;
  }

  ngOnDestroy(): void {
    const src = this.imageSrc();
    if (src) URL.revokeObjectURL(src);
  }
}
