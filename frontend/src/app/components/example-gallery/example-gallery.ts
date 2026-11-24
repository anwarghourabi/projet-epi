import { HttpClient } from '@angular/common/http';
import { Component, inject, output, signal } from '@angular/core';

interface Example { file: string; title: string; }

/** Galerie d'images d'exemple (public/examples/examples.json) : un clic lance la détection. */
@Component({
  selector: 'app-example-gallery',
  templateUrl: './example-gallery.html',
  styleUrl: './example-gallery.scss',
})
export class ExampleGallery {
  readonly picked = output<File>();
  protected readonly examples = signal<Example[]>([]);
  private readonly http = inject(HttpClient);

  constructor() {
    this.http.get<Example[]>('examples/examples.json').subscribe({
      next: (list) => this.examples.set(list),
      error: () => this.examples.set([]),
    });
  }

  protected pick(example: Example): void {
    this.http.get(`examples/${example.file}`, { responseType: 'blob' }).subscribe((blob) => {
      this.picked.emit(new File([blob], example.file, { type: blob.type || 'image/jpeg' }));
    });
  }
}
