import { DecimalPipe } from '@angular/common';
import { Component, computed, input } from '@angular/core';

import { PredictionResult, classColor } from '../../models';

/** Verdict de conformité, alertes, latences et liste des détections avec barres de score. */
@Component({
  selector: 'app-results-panel',
  templateUrl: './results-panel.html',
  styleUrl: './results-panel.scss',
  imports: [DecimalPipe],
})
export class ResultsPanel {
  readonly result = input<PredictionResult | null>(null);
  protected readonly color = classColor;

  /** Détections triées par confiance décroissante. */
  protected readonly sorted = computed(() =>
    [...(this.result()?.detections ?? [])].sort((a, b) => b.confidence - a.confidence));

  /** Nombre de détections par classe, de la plus fréquente à la moins fréquente. */
  protected readonly counts = computed(() =>
    Object.entries(this.result()?.counts ?? {}).sort((a, b) => b[1] - a[1]));
}
