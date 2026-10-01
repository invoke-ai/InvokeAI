import type { CanvasCommandRefusal } from '@workbench/canvas-engine/document/commandRefusal';
import type { SubsetOf } from '@workbench/canvas-engine/editConcurrency';
import type { LayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import type { RasterBackend, RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { SelectionState } from '@workbench/canvas-engine/selection/selectionState';
import type { Rect, Vec2 } from '@workbench/canvas-engine/types';

import { getDocumentLayer } from '@workbench/canvas-engine/document/documentIndex';
import { isEmpty, roundOut, transformBounds } from '@workbench/canvas-engine/math/rect';
import { liftSelectedPixels } from '@workbench/canvas-engine/selection/floatingSelection';
import { layerMatrix } from '@workbench/canvas-engine/tools/moveHitTest';

import type { CanvasMutationContext } from './mutationContext';

import {
  addedLayerHistoryBytes,
  layerEditRefusal,
  layerOperationStatus,
  paintLayerAt,
  publishAddedRasterLayer,
  type LayerEditRefusal,
  type LayerStepContext,
} from './editSteps';

export type NewRasterLayerResult =
  | { status: 'created'; layerId: string }
  | { status: LayerEditRefusal | SubsetOf<CanvasCommandRefusal, 'missing'> | 'empty' | 'failed' };

export interface NewRasterLayerControllerOptions {
  readonly ctx: LayerStepContext &
    Pick<CanvasMutationContext, 'begin' | 'captureInsertionAnchor' | 'createLayerId' | 'getDocument'>;
  readonly backend: RasterBackend;
  readonly layers: LayerCacheStore;
  readonly selection: SelectionState;
}

/** Shared undoable insertion for Paste and Layer via Copy, inserted above and selecting the active layer. */
export class NewRasterLayerController {
  private disposed = false;

  constructor(private readonly deps: NewRasterLayerControllerOptions) {}

  /**
   * Inserts `pixels` as a new paint layer covering `rect` (document space). The undo entry takes ownership of
   * `pixels`; the caller owns getting it into document space.
   */
  insert(rect: Rect, pixels: RasterSurface, name: string, label: string): NewRasterLayerResult {
    const { ctx } = this.deps;
    if (this.disposed) {
      return { status: 'not-ready' };
    }
    const document = ctx.getDocument();
    if (!document) {
      return { status: 'missing' };
    }
    if (isEmpty(rect)) {
      return { status: 'empty' };
    }
    const txn = ctx.begin({ historyBytes: addedLayerHistoryBytes(rect) });
    if (!('publish' in txn)) {
      return { status: layerEditRefusal(txn.status) };
    }
    try {
      const layer = paintLayerAt(ctx.createLayerId(), name, rect);
      const status = layerOperationStatus(
        publishAddedRasterLayer(ctx, txn, {
          anchor: ctx.captureInsertionAnchor('raster', document.selectedLayerId),
          label,
          layer,
          pixels,
          rect,
          selectedLayerId: document.selectedLayerId,
        })
      );
      return status === 'committed' ? { layerId: layer.id, status: 'created' } : { status };
    } finally {
      txn.end();
    }
  }

  /** Copies selected pixels into a new layer above the active one without changing the source. */
  liftSelectionToLayer(name: string, label: string): NewRasterLayerResult {
    const document = this.deps.ctx.getDocument();
    const layer = getDocumentLayer(document, document?.selectedLayerId);
    const mask = this.deps.selection.mask();
    const entry = layer ? this.deps.layers.get(layer.id) : undefined;
    if (!document || !layer || !mask || !entry || isEmpty(entry.rect)) {
      return { status: 'empty' };
    }
    const matrix = layerMatrix(layer.transform);
    const lifted = liftSelectedPixels({
      backend: this.deps.backend,
      cache: { rect: entry.rect, surface: entry.surface },
      layerMatrix: matrix,
      mask,
    });
    if (!lifted) {
      return { status: 'empty' };
    }
    // The copy comes out in LAYER-LOCAL space; a new layer is placed in document
    // space, so bake the source layer's transform into the pixels on the way out.
    const documentRect = roundOut(transformBounds(matrix, lifted.pixels.rect));
    if (isEmpty(documentRect)) {
      return { status: 'empty' };
    }
    const surface = this.deps.backend.createSurface(documentRect.width, documentRect.height);
    const ctx = surface.ctx;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.globalCompositeOperation = 'source-over';
    ctx.globalAlpha = 1;
    ctx.imageSmoothingEnabled = true;
    ctx.setTransform(matrix.a, matrix.b, matrix.c, matrix.d, matrix.e - documentRect.x, matrix.f - documentRect.y);
    ctx.drawImage(lifted.pixels.surface.canvas, lifted.pixels.rect.x, lifted.pixels.rect.y);
    ctx.setTransform(1, 0, 0, 1, 0, 0);

    return this.insert(documentRect, surface, name, label);
  }

  /** Inserts decoded clipboard pixels as a new layer, centred on `center` when given. */
  pasteImage(pixels: ImageData, name: string, label: string, center?: Vec2): NewRasterLayerResult {
    if (pixels.width <= 0 || pixels.height <= 0) {
      return { status: 'empty' };
    }
    const origin = center
      ? { x: Math.round(center.x - pixels.width / 2), y: Math.round(center.y - pixels.height / 2) }
      : { x: 0, y: 0 };
    const surface = this.deps.backend.createSurface(pixels.width, pixels.height);
    surface.ctx.setTransform(1, 0, 0, 1, 0, 0);
    surface.ctx.putImageData(pixels, 0, 0);
    return this.insert({ height: pixels.height, width: pixels.width, x: origin.x, y: origin.y }, surface, name, label);
  }

  dispose(): void {
    this.disposed = true;
  }
}
