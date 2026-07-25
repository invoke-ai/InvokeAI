import type { CanvasDocumentContractV2, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { History } from '@workbench/canvas-engine/history/history';
import type { LayerCacheStore, PreparedLayerCacheReplacement } from '@workbench/canvas-engine/render/layerCache';
import type { RasterBackend, RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { SelectionState } from '@workbench/canvas-engine/selection/selectionState';
import type { Rect, Vec2 } from '@workbench/canvas-engine/types';
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';

import { isEmpty, roundOut, transformBounds } from '@workbench/canvas-engine/math/rect';
import { liftSelectedPixels } from '@workbench/canvas-engine/selection/floatingSelection';
import { layerMatrix } from '@workbench/canvas-engine/tools/moveHitTest';

export type NewRasterLayerResult = { status: 'created'; layerId: string } | { status: 'busy' | 'empty' | 'missing' };

export interface NewRasterLayerControllerOptions {
  readonly backend: RasterBackend;
  readonly layers: LayerCacheStore;
  readonly selection: SelectionState;
  readonly history: History;
  readonly getDocument: () => CanvasDocumentContractV2 | null;
  readonly getReducerDocument: () => CanvasDocumentContractV2 | null;
  readonly capturePermit: () => object | null;
  readonly isPermitCurrent: (permit: object) => boolean;
  readonly isGestureActive: () => boolean;
  readonly endBurst: () => void;
  readonly createLayerId: () => string;
  readonly preparePixels: (layerId: string, rect: Rect, pixels: RasterSurface) => PreparedLayerCacheReplacement;
  readonly installPrepared: (prepared: PreparedLayerCacheReplacement) => void;
  readonly dispatchPrepared: (
    action: CanvasProjectMutation,
    expectedReducer: () => boolean,
    expectedMirror: () => boolean
  ) => void;
}

/**
 * Creates raster layers out of loose pixels, undoably.
 *
 * Two callers want the same thing — Paste (pixels from the system clipboard)
 * and Layer via Copy (the selection's pixels lifted off the active layer) — so
 * they share one guarded insert rather than growing two near-identical
 * controllers. The transactional shape follows `ExtractMaskedAreaController`:
 * capture a permit, build the layer, `dispatchPrepared` with reducer AND mirror
 * postconditions, install the prepared cache, then push a failure-atomic history
 * entry whose redo re-runs the exact same apply.
 *
 * The new layer is inserted directly above the active one, which is where the
 * user is looking, and becomes the selection.
 */
export class NewRasterLayerController {
  private disposed = false;

  constructor(private readonly deps: NewRasterLayerControllerOptions) {}

  /**
   * Inserts `pixels` as a new paint layer covering `rect` (document space).
   * `pixels` is consumed as-is — the caller owns getting it into document space.
   */
  insert(rect: Rect, pixels: RasterSurface, name: string, label: string): NewRasterLayerResult {
    const permit = this.deps.capturePermit();
    if (this.disposed || !permit || this.deps.isGestureActive()) {
      return { status: 'busy' };
    }
    const document = this.deps.getDocument();
    if (!document) {
      return { status: 'missing' };
    }
    if (isEmpty(rect)) {
      return { status: 'empty' };
    }
    this.deps.endBurst();

    const layerId = this.deps.createLayerId();
    // Pixels are placed by the layer SOURCE offset, not the transform, so the
    // layer's own transform stays identity — the same shape a paint layer takes
    // after a stroke, which keeps every later edit (move, transform, float) on
    // the ordinary path.
    const layer: CanvasLayerContract = {
      blendMode: 'normal',
      id: layerId,
      isEnabled: true,
      isLocked: false,
      name,
      opacity: 1,
      source: { bitmap: null, offset: { x: rect.x, y: rect.y }, type: 'paint' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'raster',
    };

    // Above the active layer: index 0 is top-most, so "above" is one index lower.
    const activeIndex = document.layers.findIndex((candidate) => candidate.id === document.selectedLayerId);
    const insertIndex = activeIndex >= 0 ? activeIndex : 0;
    const previousSelectedLayerId = document.selectedLayerId;

    const apply = (): void => {
      const prepared = this.deps.preparePixels(layerId, rect, pixels);
      this.deps.dispatchPrepared(
        {
          add: { index: insertIndex, layers: [layer] },
          enabledUpdates: [],
          selectedLayerId: layerId,
          type: 'applyCanvasLayerStackMutation',
        },
        () =>
          this.deps.getReducerDocument()?.selectedLayerId === layerId &&
          this.deps.getReducerDocument()?.layers.some((candidate) => candidate === layer) === true,
        () =>
          this.deps.getDocument()?.selectedLayerId === layerId &&
          this.deps.getDocument()?.layers.some((candidate) => candidate === layer) === true
      );
      this.deps.installPrepared(prepared);
    };

    if (!this.deps.isPermitCurrent(permit)) {
      return { status: 'busy' };
    }
    apply();
    this.deps.history.push({
      bytes: rect.width * rect.height * 4 + 256,
      label,
      redo: apply,
      replayFailureAtomic: true,
      undo: () =>
        this.deps.dispatchPrepared(
          {
            enabledUpdates: [],
            removeIds: [layerId],
            selectedLayerId: previousSelectedLayerId,
            type: 'applyCanvasLayerStackMutation',
          },
          () =>
            this.deps.getReducerDocument()?.selectedLayerId === previousSelectedLayerId &&
            this.deps.getReducerDocument()?.layers.some((candidate) => candidate.id === layerId) === false,
          () =>
            this.deps.getDocument()?.selectedLayerId === previousSelectedLayerId &&
            this.deps.getDocument()?.layers.some((candidate) => candidate.id === layerId) === false
        ),
    });
    return { layerId, status: 'created' };
  }

  /**
   * Copies the selection's pixels off the active layer into a new layer above
   * it, leaving the source untouched (Photoshop's "Layer via Copy").
   */
  liftSelectionToLayer(name: string, label: string): NewRasterLayerResult {
    const document = this.deps.getDocument();
    const layer = document?.layers.find((candidate) => candidate.id === document.selectedLayerId);
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
