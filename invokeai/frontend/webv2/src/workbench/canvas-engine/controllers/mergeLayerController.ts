import type { LayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import type { RasterBackend } from '@workbench/canvas-engine/render/raster';
import type { CanvasDocumentContractV2, CanvasLayerContract } from '@workbench/types';
import type { WorkbenchAction } from '@workbench/workbenchState';

import { mergeDownMatrix } from '@workbench/canvas-engine/document/mergeDown';
import { planMergeVisibleRuns, planNextMergeVisibleStep } from '@workbench/canvas-engine/document/mergeVisible';
import { isMergeableRasterLayer } from '@workbench/canvas-engine/document/sources';
import { isEmpty, roundOut, transformBounds, union } from '@workbench/canvas-engine/math/rect';
import { blendToComposite } from '@workbench/canvas-engine/render/compositor';

export type MergeVisibleResult = 'merged' | 'not-ready' | 'nothing';

export interface MergeLayerControllerOptions {
  readonly backend: RasterBackend;
  readonly layers: LayerCacheStore;
  readonly getDocument: () => CanvasDocumentContractV2 | null;
  readonly canEdit: () => boolean;
  readonly isGestureActive: () => boolean;
  readonly isCacheReady: (layer: CanvasLayerContract, document: CanvasDocumentContractV2) => boolean;
  readonly endBurst: () => void;
  readonly dispatch: (action: WorkbenchAction) => void;
  readonly notifyPainted: (layerId: string) => void;
  readonly markDirty: (layerId: string) => void;
}

const MAX_MERGE_VISIBLE_STEPS = 256;

/** Owns destructive merge-down and merge-visible pixel operations. */
export class MergeLayerController {
  private disposed = false;

  constructor(private readonly deps: MergeLayerControllerOptions) {}

  mergeDown(upperLayerId: string): boolean {
    if (this.disposed || !this.deps.canEdit() || this.deps.isGestureActive()) {
      return false;
    }
    this.deps.endBurst();
    const document = this.deps.getDocument();
    if (!document) {
      return false;
    }
    const upperIndex = document.layers.findIndex((layer) => layer.id === upperLayerId);
    const upper = document.layers[upperIndex];
    const below = document.layers[upperIndex + 1];
    if (upperIndex < 0 || !upper || !below || !isMergeableRasterLayer(upper) || !isMergeableRasterLayer(below)) {
      return false;
    }
    const upperCache = this.deps.layers.get(upper.id);
    const belowCache = this.deps.layers.get(below.id);
    if (
      !upperCache ||
      !belowCache ||
      !this.deps.isCacheReady(upper, document) ||
      !this.deps.isCacheReady(below, document)
    ) {
      return false;
    }
    const matrix = mergeDownMatrix(below.transform, upper.transform);
    if (!matrix) {
      return false;
    }
    if (isEmpty(belowCache.rect) && isEmpty(upperCache.rect)) {
      this.deps.dispatch({
        source: { bitmap: null, offset: { x: 0, y: 0 }, type: 'paint' },
        type: 'mergeCanvasLayersDown',
        upperLayerId,
      });
      this.deps.layers.delete(below.id);
      this.deps.notifyPainted(below.id);
      this.deps.markDirty(below.id);
      return true;
    }
    const mergedRect = roundOut(union(belowCache.rect, transformBounds(matrix, upperCache.rect)));
    const merged = this.deps.backend.createSurface(mergedRect.width, mergedRect.height);
    const context = merged.ctx;
    context.setTransform(1, 0, 0, 1, 0, 0);
    context.clearRect(0, 0, mergedRect.width, mergedRect.height);
    if (!isEmpty(belowCache.rect)) {
      context.drawImage(belowCache.surface.canvas, belowCache.rect.x - mergedRect.x, belowCache.rect.y - mergedRect.y);
    }
    if (!isEmpty(upperCache.rect)) {
      context.setTransform(matrix.a, matrix.b, matrix.c, matrix.d, matrix.e - mergedRect.x, matrix.f - mergedRect.y);
      context.globalAlpha = upper.opacity;
      context.globalCompositeOperation = blendToComposite(upper.blendMode);
      context.drawImage(upperCache.surface.canvas, upperCache.rect.x, upperCache.rect.y);
      context.setTransform(1, 0, 0, 1, 0, 0);
      context.globalAlpha = 1;
      context.globalCompositeOperation = 'source-over';
    }
    this.deps.dispatch({
      source: { bitmap: null, offset: { x: mergedRect.x, y: mergedRect.y }, type: 'paint' },
      type: 'mergeCanvasLayersDown',
      upperLayerId,
    });
    this.deps.layers.delete(below.id);
    const target = this.deps.layers.getOrCreateRect(below.id, mergedRect);
    target.surface.ctx.drawImage(merged.canvas, 0, 0);
    target.stale = false;
    this.deps.notifyPainted(below.id);
    this.deps.markDirty(below.id);
    return true;
  }

  mergeVisible(): MergeVisibleResult {
    if (this.disposed || !this.deps.canEdit() || this.deps.isGestureActive()) {
      return 'nothing';
    }
    const document = this.deps.getDocument();
    if (!document) {
      return 'nothing';
    }
    const runs = planMergeVisibleRuns(document.layers);
    if (runs.length === 0) {
      return 'nothing';
    }
    const byId = new Map(document.layers.map((layer) => [layer.id, layer]));
    for (const run of runs) {
      for (const id of run) {
        const layer = byId.get(id);
        if (!layer || !this.deps.isCacheReady(layer, document)) {
          return 'not-ready';
        }
      }
      for (let index = 0; index + 1 < run.length; index += 1) {
        const upper = byId.get(run[index] ?? '');
        const below = byId.get(run[index + 1] ?? '');
        if (!upper || !below || !mergeDownMatrix(below.transform, upper.transform)) {
          return 'not-ready';
        }
      }
    }
    for (let step = 0; step < MAX_MERGE_VISIBLE_STEPS; step += 1) {
      const liveDocument = this.deps.getDocument();
      if (!liveDocument) {
        break;
      }
      const next = planNextMergeVisibleStep(liveDocument.layers);
      if (!next) {
        break;
      }
      if (next.orderedIds) {
        this.deps.dispatch({ orderedIds: next.orderedIds, type: 'reorderCanvasLayers' });
      }
      if (!this.mergeDown(next.upperId)) {
        break;
      }
    }
    return 'merged';
  }

  dispose(): void {
    this.disposed = true;
  }
}
