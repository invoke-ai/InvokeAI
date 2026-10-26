import type { ControlLayerViewMode, ControlLayerViewOutcome } from '@workbench/canvas-engine/capabilities';
import type {
  CanvasControlLayerContract,
  CanvasDocumentContractV3,
  CanvasPixelSourceContract,
} from '@workbench/canvas-engine/contracts';
import type { ControlLayerView } from '@workbench/canvas-engine/engineStores';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { RasterizeResult } from '@workbench/canvas-engine/render/rasterizers';
import type { Mat2d, Rect } from '@workbench/canvas-engine/types';

import { getDocumentLayer } from '@workbench/canvas-engine/document/documentIndex';
import { IDENTITY_TRANSFORM, layerTransformMatrix } from '@workbench/canvas-engine/transform/transformMath';

import type { RasterMemoryLease, RasterMemoryReservationResult } from './rasterMemoryBudgetController';

/** A shown view: the kept original's pixels, placed in the layer-local frame of the map. */
export interface ControlLayerViewState extends Readonly<ControlLayerView> {
  readonly surface: RasterSurface;
  /** The original's pixels in its own local frame. */
  readonly rect: Rect;
  /** Original-local → layer-local (`originalTransform`). */
  readonly matrix: Mat2d;
  /** The contract the view was built from; any other identity means the view no longer describes the layer. */
  readonly layer: CanvasControlLayerContract;
}

export interface ControlViewControllerOptions {
  readonly getDocument: () => CanvasDocumentContractV3 | null;
  readonly rasterize: (
    source: CanvasPixelSourceContract,
    document: CanvasDocumentContractV3,
    signal: AbortSignal
  ) => Promise<RasterizeResult>;
  /** Reserves raster memory for the original's surface while it is decoded. */
  readonly reserve: (bytes: number) => RasterMemoryReservationResult;
  /** Accounts the decoded surface as retained pixels until the view releases it. */
  readonly trackDetached: (bytes: number) => RasterMemoryLease;
  /** Whether an operation or a generation holds the canvas; views are refused meanwhile. */
  readonly isLocked: () => boolean;
  readonly invalidateLayer: (layerId: string) => void;
  readonly publish: (view: ControlLayerView | null) => void;
}

const asControlLayer = (
  document: CanvasDocumentContractV3 | null,
  layerId: string
): CanvasControlLayerContract | null => {
  const layer = getDocumentLayer(document, layerId);
  return layer?.type === 'control' ? layer : null;
};

/** RGBA bytes of the original's decoded pixels; null when it has none. */
const originalBytes = (original: CanvasPixelSourceContract): number | null => {
  const image = original.type === 'image' ? original.image : original.bitmap;
  return image && image.width > 0 && image.height > 0 ? image.width * image.height * 4 : null;
};

/**
 * Owns the display-only control-layer view: at most one layer shows its kept original (alone, or under its map),
 * decoded under a raster reservation and retained as detached pixels until the view ends. A newer request supersedes
 * an older decode. Nothing here touches the document, history or exports; callers end the view whenever the layer
 * stops matching it.
 */
export class ControlViewController {
  private state: ControlLayerViewState | null = null;
  private lease: RasterMemoryLease | null = null;
  private pending: { readonly layerId: string; readonly abort: AbortController } | null = null;

  constructor(private readonly options: ControlViewControllerOptions) {}

  get(): ControlLayerViewState | null {
    return this.state;
  }

  async set(layerId: string, view: ControlLayerViewMode): Promise<ControlLayerViewOutcome> {
    this.pending?.abort.abort();
    this.pending = null;
    if (view === 'map') {
      this.clear((id) => id === layerId);
      return 'shown';
    }
    if (this.options.isLocked()) {
      return 'locked';
    }
    const document = this.options.getDocument();
    const layer = asControlLayer(document, layerId);
    const bytes = layer?.original ? originalBytes(layer.original) : null;
    if (!document || !layer?.original || bytes === null) {
      return 'missing';
    }
    // Original and Overlay share one decoded surface.
    if (this.state?.layer === layer) {
      this.show({ ...this.state, view }, this.lease);
      return 'shown';
    }
    const reservation = this.options.reserve(bytes);
    if (reservation.status !== 'ok') {
      return 'over-budget';
    }
    const pending = { abort: new AbortController(), layerId };
    this.pending = pending;
    try {
      const { rect, surface } = await this.options.rasterize(layer.original, document, pending.abort.signal);
      if (this.pending !== pending || asControlLayer(this.options.getDocument(), layerId) !== layer) {
        return 'stale';
      }
      const matrix = layerTransformMatrix(layer.originalTransform ?? IDENTITY_TRANSFORM);
      this.show({ layer, layerId, matrix, rect, surface, view }, this.options.trackDetached(bytes));
      return 'shown';
    } catch {
      // The map keeps showing either way.
      return this.pending === pending ? 'failed' : 'stale';
    } finally {
      reservation.lease.release();
      if (this.pending === pending) {
        this.pending = null;
      }
    }
  }

  /** Ends views and in-flight requests for the layers `which` names; every layer when omitted. */
  clear(which: (layerId: string) => boolean = () => true): void {
    if (this.pending && which(this.pending.layerId)) {
      this.pending.abort.abort();
      this.pending = null;
    }
    const state = this.state;
    if (!state || !which(state.layerId)) {
      return;
    }
    this.state = null;
    this.lease?.release();
    this.lease = null;
    this.options.publish(null);
    this.options.invalidateLayer(state.layerId);
  }

  /** Ends the view once its layer is gone or is no longer the contract it was built from. */
  reconcile(): void {
    const state = this.state;
    if (state && asControlLayer(this.options.getDocument(), state.layerId) !== state.layer) {
      this.clear();
    }
  }

  dispose(): void {
    this.clear();
  }

  private show(next: ControlLayerViewState, lease: RasterMemoryLease | null): void {
    const previous = this.state;
    if (previous && previous.layerId !== next.layerId) {
      this.options.invalidateLayer(previous.layerId);
    }
    if (this.lease !== lease) {
      this.lease?.release();
    }
    this.state = next;
    this.lease = lease;
    this.options.publish({ layerId: next.layerId, view: next.view });
    this.options.invalidateLayer(next.layerId);
  }
}
