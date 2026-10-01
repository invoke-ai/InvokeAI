import type { CanvasDocumentContractV3, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { History } from '@workbench/canvas-engine/history/history';
import type { CanvasProjectMutation } from '@workbench/canvas-engine/mutationContracts';
import type { LayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import type { RasterBackend } from '@workbench/canvas-engine/render/raster';
import type { RasterizeDeps } from '@workbench/canvas-engine/render/rasterizers';

import { getDocumentLayer } from '@workbench/canvas-engine/document/documentIndex';
import { getSourceContentRect, isEmptyPolygonShape } from '@workbench/canvas-engine/document/sources';
import { collectHistoryMediaRefs } from '@workbench/canvas-engine/history/history';
import { roundOut, transformBounds } from '@workbench/canvas-engine/math/rect';
import { rasterizeSource } from '@workbench/canvas-engine/render/rasterizers';
import { bakeMatrix } from '@workbench/canvas-engine/transform/transformMath';

export interface RasterizeLayerControllerOptions {
  readonly backend: RasterBackend;
  readonly layers: LayerCacheStore;
  readonly history: History;
  readonly getDocument: () => CanvasDocumentContractV3 | null;
  readonly rasterizeDeps: (document: CanvasDocumentContractV3) => RasterizeDeps;
  readonly dispatch: (action: CanvasProjectMutation) => void;
  readonly canEdit: () => boolean;
  readonly isGestureActive: () => boolean;
  readonly endBurst: () => void;
  readonly notifyPainted: (layerId: string) => void;
  readonly markDirty: (layerId: string) => void;
}

/** Owns conversion of parametric raster layers into persisted paint pixels. */
export class RasterizeLayerController {
  private disposed = false;

  constructor(private readonly deps: RasterizeLayerControllerOptions) {}

  rasterize(layerId: string): boolean {
    if (this.disposed || !this.deps.canEdit() || this.deps.isGestureActive()) {
      return false;
    }
    this.deps.endBurst();
    const document = this.deps.getDocument();
    const layer = getDocumentLayer(document, layerId);
    if (!document || !layer || layer.type !== 'raster' || layer.isLocked) {
      return false;
    }
    const source = layer.source;
    if (
      (source.type !== 'shape' && source.type !== 'gradient' && source.type !== 'text') ||
      (source.type === 'shape' && isEmptyPolygonShape(source))
    ) {
      return false;
    }
    const parametricLayer: CanvasLayerContract = structuredClone(layer);
    const apply = (): void => {
      const liveDocument = this.deps.getDocument();
      const liveLayer = getDocumentLayer(liveDocument, layerId);
      if (!liveDocument || !liveLayer || liveLayer.type !== 'raster') {
        return;
      }
      const liveSource = liveLayer.source;
      if (liveSource.type !== 'shape' && liveSource.type !== 'gradient' && liveSource.type !== 'text') {
        return;
      }
      const contentRect = getSourceContentRect(liveLayer, liveDocument);
      const scratch = this.deps.backend.createSurface(contentRect.width, contentRect.height);
      void rasterizeSource(liveSource, this.deps.rasterizeDeps(liveDocument), scratch);
      const matrix = bakeMatrix(liveLayer.transform);
      const bakedRect = roundOut(transformBounds(matrix, contentRect));
      const baked = this.deps.backend.createSurface(bakedRect.width, bakedRect.height);
      baked.ctx.setTransform(1, 0, 0, 1, 0, 0);
      baked.ctx.clearRect(0, 0, bakedRect.width, bakedRect.height);
      baked.ctx.imageSmoothingEnabled = true;
      baked.ctx.setTransform(matrix.a, matrix.b, matrix.c, matrix.d, matrix.e - bakedRect.x, matrix.f - bakedRect.y);
      baked.ctx.drawImage(scratch.canvas, contentRect.x, contentRect.y);
      baked.ctx.setTransform(1, 0, 0, 1, 0, 0);
      const paintLayer: CanvasLayerContract = {
        ...liveLayer,
        source: { bitmap: null, offset: { x: bakedRect.x, y: bakedRect.y }, type: 'paint' },
        transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      };
      this.deps.dispatch({ id: layerId, layer: paintLayer, targetType: 'raster', type: 'convertCanvasLayer' });
      this.deps.layers.delete(layerId);
      const target = this.deps.layers.getOrCreateRect(layerId, bakedRect);
      target.surface.ctx.drawImage(baked.canvas, 0, 0);
      target.stale = false;
      this.deps.notifyPainted(layerId);
      this.deps.markDirty(layerId);
    };
    apply();
    this.deps.history.push({
      bytes: 256,
      heldAssetRefs: collectHistoryMediaRefs(parametricLayer),
      label: 'Rasterize layer',
      redo: apply,
      undo: () =>
        this.deps.dispatch({ id: layerId, layer: parametricLayer, targetType: 'raster', type: 'convertCanvasLayer' }),
    });
    return true;
  }

  dispose(): void {
    this.disposed = true;
  }
}
