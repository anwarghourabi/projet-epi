import { Component, output, signal } from '@angular/core';

const ACCEPTED = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];

/** Zone de dépôt : glisser-déposer ou clic pour choisir une image. */
@Component({
  selector: 'app-upload-zone',
  templateUrl: './upload-zone.html',
  styleUrl: './upload-zone.scss',
})
export class UploadZone {
  readonly fileSelected = output<File>();
  readonly invalidFile = output<string>();
  protected readonly dragging = signal(false);

  protected onDragOver(event: DragEvent): void {
    event.preventDefault();
    this.dragging.set(true);
  }

  protected onDrop(event: DragEvent): void {
    event.preventDefault();
    this.dragging.set(false);
    const file = event.dataTransfer?.files?.[0];
    if (file) this.handle(file);
  }

  protected onChange(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (file) this.handle(file);
    input.value = ''; // permet de rechoisir le même fichier
  }

  private handle(file: File): void {
    if (!ACCEPTED.includes(file.type)) {
      this.invalidFile.emit(`« ${file.name} » n'est pas une image (formats acceptés : JPEG, PNG, WebP, BMP).`);
      return;
    }
    this.fileSelected.emit(file);
  }
}
