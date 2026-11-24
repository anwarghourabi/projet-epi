import { Component, ElementRef, OnDestroy, input, output, signal, viewChild } from '@angular/core';

/** Webcam : capture une image toutes les `intervalMs` ms et l'émet (si l'API n'est pas occupée). */
@Component({
  selector: 'app-webcam',
  templateUrl: './webcam.html',
  styleUrl: './webcam.scss',
})
export class Webcam implements OnDestroy {
  readonly busy = input(false);
  readonly intervalMs = input(700);
  readonly frame = output<Blob>();
  readonly cameraError = output<string>();

  protected readonly running = signal(false);
  private readonly video = viewChild.required<ElementRef<HTMLVideoElement>>('video');
  private stream: MediaStream | null = null;
  private timer: ReturnType<typeof setInterval> | null = null;

  async start(): Promise<void> {
    if (!navigator.mediaDevices?.getUserMedia) {
      this.cameraError.emit("Ce navigateur ne donne pas accès à la caméra (utilisez Chrome/Edge, en http://localhost).");
      return;
    }
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: false });
    } catch (err) {
      const name = (err as DOMException).name;
      this.cameraError.emit(name === 'NotAllowedError'
        ? "Accès à la caméra refusé : autorisez-le dans la barre d'adresse du navigateur."
        : name === 'NotFoundError' ? 'Aucune caméra détectée sur cet ordinateur.'
        : `Impossible d'ouvrir la caméra (${name}).`);
      return;
    }
    const video = this.video().nativeElement;
    video.srcObject = this.stream;
    await video.play();
    this.running.set(true);
    this.timer = setInterval(() => this.capture(), this.intervalMs());
  }

  stop(): void {
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    this.running.set(false);
  }

  ngOnDestroy(): void {
    this.stop();
  }

  /** Copie l'image courante de la vidéo dans un canvas puis en JPEG. */
  private capture(): void {
    const video = this.video().nativeElement;
    if (this.busy() || !video.videoWidth) return; // on attend la réponse précédente
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d')!.drawImage(video, 0, 0);
    canvas.toBlob((blob) => blob && this.frame.emit(blob), 'image/jpeg', 0.85);
  }
}
