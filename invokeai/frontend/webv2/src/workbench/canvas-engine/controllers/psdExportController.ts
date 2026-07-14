import type { RasterBackend, RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Rect } from '@workbench/canvas-engine/types';
import type { CanvasDocumentContractV2, CanvasLayerContract } from '@workbench/types';

import { getSourceContentRect } from '@workbench/canvas-engine/document/sources';
import { executePsdExport, planPsdExport, type PsdExportLayerInput } from '@workbench/canvas-engine/export/psdExport';
import { isEmpty } from '@workbench/canvas-engine/math/rect';

export type PsdExportResult = 'exported' | 'nothing' | 'too-large' | 'not-ready';

export interface PsdExportControllerOptions {
  readonly backend: RasterBackend;
  readonly getDocument: () => CanvasDocumentContractV2 | null;
  readonly isRasterizationCurrent: (layer: CanvasLayerContract) => boolean;
  readonly getLayerSurface: (layerId: string) => Promise<{ surface: RasterSurface; rect: Rect }>;
}

const isExportable = (layer: CanvasLayerContract): boolean => {
  if (layer.type !== 'raster') {
    return false;
  }
  switch (layer.source.type) {
    case 'paint':
    case 'image':
    case 'gradient':
    case 'text':
      return true;
    case 'shape':
      return layer.source.kind !== 'polygon';
    default:
      return false;
  }
};

/** Owns PSD export planning, readiness preflight, and execution. */
export class PsdExportController {
  private disposed = false;

  constructor(private readonly deps: PsdExportControllerOptions) {}

  async export(fileName: string): Promise<PsdExportResult> {
    if (this.disposed) {
      return 'nothing';
    }
    const document = this.deps.getDocument();
    if (!document) {
      return 'nothing';
    }
    const layers = document.layers.filter(isExportable);
    if (layers.length === 0) {
      return 'nothing';
    }
    for (const layer of layers) {
      if (!isEmpty(getSourceContentRect(layer, document)) && this.deps.isRasterizationCurrent(layer)) {
        return 'not-ready';
      }
    }
    const inputs: PsdExportLayerInput[] = layers.map((layer) => ({
      adjustments: layer.type === 'raster' ? layer.adjustments : undefined,
      blendMode: layer.blendMode,
      contentRect: getSourceContentRect(layer, document),
      id: layer.id,
      isEnabled: layer.isEnabled,
      name: layer.name,
      opacity: layer.opacity,
      transform: layer.transform,
    }));
    const plan = planPsdExport(inputs);
    if (plan.status === 'empty') {
      return 'nothing';
    }
    if (plan.status === 'too-large') {
      return 'too-large';
    }
    await executePsdExport(plan, /\.psd$/i.test(fileName) ? fileName : `${fileName}.psd`, {
      backend: this.deps.backend,
      getLayerSurface: this.deps.getLayerSurface,
    });
    return 'exported';
  }

  dispose(): void {
    this.disposed = true;
  }
}
