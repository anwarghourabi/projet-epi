import { Component, ElementRef, effect, input, viewChild } from '@angular/core';

import { Detection, classColor } from '../../models';

/** Affiche l'image et dessine les boîtes renvoyées par l'API sur un canvas. */
@Component({
  selector: 'app-detection-viewer',
  templateUrl: './detection-viewer.html',
  styleUrl: './detection-viewer.scss',
})
export class DetectionViewer {
  readonly src = input<string | null>(null);
  readonly detections = input<Detection[]>([]);
  readonly loading = input(false);
  private readonly canvas = viewChild.required<ElementRef<HTMLCanvasElement>>('canvas');

  constructor() {
    // Redessine dès que l'image ou les détections changent
    effect(() => {
      const src = this.src();
      const detections = this.detections();
      if (!src) return;
      const img = new Image();
      img.onload = () => this.draw(img, detections);
      img.src = src;
    });
  }

  private draw(img: HTMLImageElement, detections: Detection[]): void {
    const canvas = this.canvas().nativeElement;
    // Le canvas a la taille réelle de l'image : les coordonnées de l'API s'appliquent telles quelles
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const ctx = canvas.getContext('2d')!;
    ctx.drawImage(img, 0, 0);

    const line = Math.max(2, Math.round(Math.max(canvas.width, canvas.height) / 400));
    const fontSize = Math.max(12, Math.round(Math.max(canvas.width, canvas.height) / 60));
    ctx.font = `600 ${fontSize}px system-ui, sans-serif`;
    ctx.textBaseline = 'top';

    for (const d of detections) {
      const { x1, y1, x2, y2 } = d.bbox;
      const color = classColor(d.class_name);
      ctx.strokeStyle = color;
      ctx.lineWidth = line;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      const label = `${d.class_name} ${(d.confidence * 100).toFixed(0)}%`;
      const w = ctx.measureText(label).width + 8;
      const h = fontSize + 6;
      const y = y1 - h >= 0 ? y1 - h : y1; // étiquette au-dessus de la boîte si la place le permet
      ctx.fillStyle = color;
      ctx.fillRect(x1, y, w, h);
      ctx.fillStyle = '#fff';
      ctx.fillText(label, x1 + 4, y + 3);
    }
  }
}
