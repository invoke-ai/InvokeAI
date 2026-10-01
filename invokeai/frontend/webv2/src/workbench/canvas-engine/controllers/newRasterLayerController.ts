import type { CanvasDocumentContractV3, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { CanvasCommandRefusal } from '@workbench/canvas-engine/document/commandRefusal';
import type { CanvasNodeInsertionAnchor } from '@workbench/canvas-engine/document/insertionAnchors';
import type { CanvasTransactionOutcome, SubsetOf } from '@workbench/canvas-engine/editConcurrency';
import type { HeldAssetRefs } from '@workbench/canvas-engine/history/history';
import type { CanvasProjectMutation } from '@workbench/canvas-engine/mutationContracts';
import type { LayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import type { RasterBackend, RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { SelectionState } from '@workbench/canvas-engine/selection/selectionState';
import type { Rect, Vec2 } from '@workbench/canvas-engine/types';

import { getDocumentLayer, isNodeAbsent } from '@workbench/canvas-engine/document/documentIndex';
import { collectHistoryMediaRefs, HISTORY_ENTRY_OVERHEAD_BYTES } from '@workbench/canvas-engine/history/history';
import { isEmpty, roundOut, transformBounds } from '@workbench/canvas-engine/math/rect';
import { liftSelectedPixels } from '@workbench/canvas-engine/selection/floatingSelection';
import { layerMatrix } from '@workbench/canvas-engine/tools/moveHitTest';

import type {
  CanvasMutationContext,
  EditPublishResult,
  EditRefusal,
  EditStep,
  EditTransaction,
} from './mutationContext';

/** A refused admission as its caller reports it; a gesture in progress reads as contention. */
export type LayerEditRefusal = SubsetOf<CanvasTransactionOutcome, 'busy' | 'not-ready' | 'over-budget'>;

/** How a layer operation's publication ended; a reducer refusal or failed postcondition applied nothing. */
export type LayerEditStatus = 'committed' | LayerEditRefusal | 'failed';

export const layerEditRefusal = (status: EditRefusal): LayerEditRefusal =>
  status === 'gesture-active' ? 'busy' : status;

export const layerEditStatus = (result: EditPublishResult): LayerEditStatus => {
  switch (result.status) {
    case 'committed':
      return 'committed';
    case 'dispatch-rejected':
    case 'postcondition-failed':
      return 'failed';
    default:
      return layerEditRefusal(result.status);
  }
};

export const rgbaBytes = (rect: Rect): number => Math.max(0, rect.width) * Math.max(0, rect.height) * 4;

export type ReplayContext = Pick<CanvasMutationContext, 'reserveRaster'>;

/** Runs a replay's preparation under a raster reservation; a replay that does not fit throws and stays in place. */
export const withReplayReservation = <T>(ctx: ReplayContext, bytes: number, run: () => T): T => {
  const lease = ctx.reserveRaster(bytes);
  if (!lease) {
    throw new Error('Not enough raster memory to replay this step.');
  }
  try {
    return run();
  } finally {
    lease.release();
  }
};

type EnabledUpdates = readonly { readonly id: string; readonly isEnabled: boolean }[];

/** A new paint layer whose pixels the undo entry owns, inserted and selected in one step. */
export interface AddedRasterLayerEdit {
  readonly label: string;
  /** A raster paint layer whose source offset places `pixels` at `rect`. */
  readonly layer: CanvasLayerContract;
  readonly rect: Rect;
  readonly pixels: RasterSurface;
  readonly anchor: CanvasNodeInsertionAnchor;
  /** The selection undo restores. */
  readonly selectedLayerId: string | null;
  /** Enabled flags the edit sets on other layers, and the values undo restores. */
  readonly enabled?: { readonly before: EnabledUpdates; readonly after: EnabledUpdates };
  readonly heldAssetRefs?: HeldAssetRefs;
}

export type AddedLayerContext = Pick<CanvasMutationContext, 'applyStep' | 'installPrepared' | 'preparePixels'> &
  ReplayContext;

/** Bytes an {@link AddedRasterLayerEdit} entry retains. */
export const addedLayerHistoryBytes = (rect: Rect): number => rgbaBytes(rect) + HISTORY_ENTRY_OVERHEAD_BYTES;

/**
 * Inserts the layer with its prepared pixels as one undo step. The cache replacement is reserved and prepared
 * before the document changes; replays prepare again under their own reservation.
 */
export const publishAddedRasterLayer = (
  ctx: AddedLayerContext,
  txn: EditTransaction,
  edit: AddedRasterLayerEdit
): EditPublishResult => {
  const { anchor, layer, pixels, rect, selectedLayerId } = edit;
  const after = edit.enabled?.after ?? [];
  const before = edit.enabled?.before ?? [];
  const forward: CanvasProjectMutation = {
    add: [{ anchor, nodes: [layer] }],
    enabledUpdates: after,
    selectedLayerId: layer.id,
    type: 'applyCanvasLayerStackMutation',
  };
  const inverse: CanvasProjectMutation = {
    enabledUpdates: before,
    removeIds: [layer.id],
    selectedLayerId,
    type: 'applyCanvasLayerStackMutation',
  };
  const hasEnabled = (document: CanvasDocumentContractV3 | null, updates: EnabledUpdates): boolean =>
    updates.every((update) => getDocumentLayer(document, update.id)?.isEnabled === update.isEnabled);
  const added = (document: CanvasDocumentContractV3 | null): boolean =>
    document?.selectedLayerId === layer.id &&
    getDocumentLayer(document, layer.id) === layer &&
    hasEnabled(document, after);
  const removed = (document: CanvasDocumentContractV3 | null): boolean =>
    document !== null &&
    document.selectedLayerId === selectedLayerId &&
    isNodeAbsent(document, layer.id) &&
    hasEnabled(document, before);
  const addStep = (): EditStep => {
    const prepared = ctx.preparePixels(layer.id, rect, pixels);
    return {
      accepted: added,
      install: () => ctx.installPrepared(prepared),
      mutation: forward,
      rollback: { mutation: inverse, restored: removed },
    };
  };
  if (!txn.reserveRaster(rgbaBytes(rect))) {
    return { status: 'over-budget' };
  }
  return txn.publish(edit.label, addStep(), {
    bytes: addedLayerHistoryBytes(rect),
    heldAssetRefs: edit.heldAssetRefs ?? collectHistoryMediaRefs(layer),
    redo: () => withReplayReservation(ctx, rgbaBytes(rect), () => ctx.applyStep(addStep())),
    undo: () =>
      ctx.applyStep({ accepted: removed, mutation: inverse, rollback: { mutation: forward, restored: added } }),
  });
};

/** A raster paint layer placing pixels at `rect` with an identity transform. */
export const paintLayerAt = (id: string, name: string, rect: Rect): CanvasLayerContract => ({
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  name,
  opacity: 1,
  source: { bitmap: null, offset: { x: rect.x, y: rect.y }, type: 'paint' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'raster',
});

export type NewRasterLayerResult =
  | { status: 'created'; layerId: string }
  | { status: LayerEditRefusal | SubsetOf<CanvasCommandRefusal, 'missing'> | 'empty' | 'failed' };

export interface NewRasterLayerControllerOptions {
  readonly ctx: AddedLayerContext &
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
      const status = layerEditStatus(
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
