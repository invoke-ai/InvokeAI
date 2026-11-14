import type {
  BeginTransformResult,
  CanvasEditRefusal,
  StructuralCommitResult,
} from '@workbench/canvas-engine/capabilities';
import type { CanvasDocumentContractV3, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { TransformOptions, TransformSession } from '@workbench/canvas-engine/engineStores';
import type { CanvasProjectMutation } from '@workbench/canvas-engine/mutationContracts';
import type { LayerCacheEntry } from '@workbench/canvas-engine/render/layerCache';
import type { RasterBackend, RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { LayerTransform } from '@workbench/canvas-engine/transform/transformMath';
import type { Rect, ToolId } from '@workbench/canvas-engine/types';

import { lookupDocumentLeaf } from '@workbench/canvas-engine/document-model/documentModel';
import { rebaseControlOriginal } from '@workbench/canvas-engine/document/controlDerivation';
import { getDocumentLayer } from '@workbench/canvas-engine/document/documentIndex';
import { isLeafEditable } from '@workbench/canvas-engine/document/layerEligibility';
import { isRenderableLayer } from '@workbench/canvas-engine/document/sources';
import {
  collectHistoryMediaRefs,
  HISTORY_ENTRY_OVERHEAD_BYTES,
  NO_HELD_ASSET_REFS,
} from '@workbench/canvas-engine/history/history';
import { isEmpty, roundOut, transformBounds } from '@workbench/canvas-engine/math/rect';
import { hasSpatialAdjustments, isPositionDependentAdjustments } from '@workbench/canvas-engine/render/adjustments';
import {
  adjustExactly,
  adjustmentRefusal,
  imageDataOf,
  type SpatialRunner,
} from '@workbench/canvas-engine/render/exactAdjustments';
import { hittableLayerSize } from '@workbench/canvas-engine/tools/moveHitTest';
import { isTransformEligible } from '@workbench/canvas-engine/tools/transformTool';
import { bakeMatrix } from '@workbench/canvas-engine/transform/transformMath';

import type { CanvasMutationContext, EditPublishResult, EditStep } from './mutationContext';

import { replaceLayerStep, rgbaBytes, withReplayReservation } from './editSteps';

export type TransformResampler = (
  surface: RasterSurface,
  target: { width: number; height: number },
  mode: TransformOptions['resampleMode'],
  signal: AbortSignal,
  onProgress?: (phase: 'queued' | 'resampling') => void
) => Promise<ImageData>;

export interface TransformEditingControllerOptions {
  readonly getOptions?: () => TransformOptions;
  readonly resample?: TransformResampler;
  readonly runSpatial?: SpatialRunner;
  readonly reportFailure?: (layerId: string, error: unknown) => void;
  readonly session: { get(): TransformSession | null; set(value: TransformSession | null): void };
  /** A requested session waiting for its layer's pixels to persist; see {@link TransformEditingController.request}. */
  readonly pending: { get(): string | null; set(value: string | null): void };
  readonly backend: RasterBackend;
  readonly ctx: Pick<
    CanvasMutationContext,
    'applyStep' | 'begin' | 'installPrepared' | 'preparePixels' | 'reserveRaster'
  >;
  readonly getDocument: () => CanvasDocumentContractV3 | null;
  readonly getCache: (layerId: string) => LayerCacheEntry | null;
  /** The live cache extent, read without touching residency order. */
  readonly peekCacheRect: (layerId: string) => Rect | undefined;
  readonly setOverride: (layerId: string, transform: LayerTransform | null) => void;
  /** A paint layer's pixels were redrawn from `from`, the session's transform, into its new local space. */
  readonly rebaseLineEnd: (layerId: string, from: LayerTransform) => void;
  readonly commitStructural: (
    label: string,
    forward: CanvasProjectMutation,
    inverse: CanvasProjectMutation
  ) => StructuralCommitResult;
  readonly reportRefusal: (refusal: CanvasEditRefusal) => void;
  readonly canEdit: () => boolean;
  readonly isGestureActive: () => boolean;
  readonly hasFloatingSelection: () => boolean;
  /** The tool the user is on: a Space, Alt or C hold only lends the surface to another tool. */
  readonly getUserToolId: () => ToolId;
  /** Selects `layerId`; selecting cancels a session on another layer. */
  readonly select: (layerId: string) => void;
  /** Switches to the transform tool, which opens a session on an eligible selection. */
  readonly enterTool: () => void;
  readonly invalidate: (payload: { layers: string[]; overlay: true }) => void;
}

const IDENTITY: LayerTransform = { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 };

/**
 * What a bake step changes on the layer: only its transform, or the whole contract when more than the transform moves
 * with the pixels (a control layer's kept original).
 */
type BakeChange =
  | { readonly kind: 'transform'; readonly from: LayerTransform; readonly to: LayerTransform }
  | { readonly kind: 'contract'; readonly from: CanvasLayerContract; readonly to: CanvasLayerContract };

const reversed = (change: BakeChange): BakeChange =>
  change.kind === 'transform'
    ? { from: change.to, kind: 'transform', to: change.from }
    : { from: change.to, kind: 'contract', to: change.from };

const isRefusal = (status: string): status is Extract<CanvasEditRefusal, string> =>
  status === 'busy' || status === 'gesture-active' || status === 'not-ready' || status === 'over-budget';

const unchanged = (left: LayerTransform, right: LayerTransform): boolean =>
  left.x === right.x &&
  left.y === right.y &&
  left.scaleX === right.scaleX &&
  left.scaleY === right.scaleY &&
  left.rotation === right.rotation;

/**
 * Owns transform session state, previews, param commits, paint bakes, and the wait of a session requested on pixels
 * persistence has not acknowledged yet.
 */
export class TransformEditingController {
  private disposed = false;
  private publishingResampled = false;
  private processing: { abort: AbortController; layer: CanvasLayerContract } | null = null;

  constructor(private readonly deps: TransformEditingControllerOptions) {}

  begin(layerId: string): void {
    if (this.disposed) {
      return;
    }
    const document = this.deps.getDocument();
    const leaf = lookupDocumentLeaf(document, layerId);
    const layer = leaf?.layer;
    if (!document || !layer || !isLeafEditable(leaf) || !hittableLayerSize(layer, document)) {
      return;
    }
    this.cancel();
    const start = { ...layer.transform };
    this.deps.session.set({ layerId, startTransform: start, transform: start });
    this.deps.setOverride(layerId, start);
    this.deps.invalidate({ layers: [layerId], overlay: true });
  }

  /**
   * Opens a session on `layerId` now, whatever tool is active: selects it, enters the tool and opens the session
   * even when transform is already active, cancelling a session on another layer. Every refusal is decided before
   * the selection or tool changes. A leaf whose only content is a stroke persistence has not acknowledged yet (a new
   * paint layer's first upload) has no persisted extent to frame, so the request waits in `pending` and the session
   * opens once the bitmap lands.
   */
  request(layerId: string): BeginTransformResult {
    const document = this.deps.getDocument();
    const leaf = lookupDocumentLeaf(document, layerId);
    if (this.disposed || !document || !leaf) {
      return this.disposed ? 'busy' : 'missing';
    }
    if (leaf.effectiveLocked) {
      return 'locked';
    }
    const persisted = isTransformEligible(leaf.layer, document);
    if (!persisted && !isTransformEligible(leaf.layer, document, this.deps.peekCacheRect(layerId))) {
      return 'ineligible';
    }
    if (this.deps.isGestureActive() || this.deps.hasFloatingSelection() || !this.deps.canEdit()) {
      return 'busy';
    }
    if (document.selectedLayerId !== layerId) {
      this.deps.select(layerId);
    }
    this.deps.enterTool();
    if (this.deps.session.get()?.layerId !== layerId) {
      this.cancel();
      if (persisted) {
        this.begin(layerId);
      }
      if (this.deps.session.get()?.layerId !== layerId) {
        this.deps.pending.set(layerId);
      }
    }
    return 'started';
  }

  /**
   * Every document change settles a wait: the session opens once the layer's pixels are persisted, if it is still
   * the selection, transform is the tool the user is on and no session opened meanwhile; otherwise the wait ends.
   */
  onDocumentChanged(): void {
    if (this.processing) {
      const document = this.deps.getDocument();
      const layer = this.processing.layer;
      if (
        (!this.publishingResampled && getDocumentLayer(document, layer.id) !== layer) ||
        document?.selectedLayerId !== layer.id ||
        !isLeafEditable(lookupDocumentLeaf(document, layer.id))
      ) {
        this.cancel();
      }
    }
    const layerId = this.deps.pending.get();
    if (layerId === null) {
      return;
    }
    const document = this.deps.getDocument();
    const layer = getDocumentLayer(document, layerId);
    if (
      !document ||
      !layer ||
      document.selectedLayerId !== layerId ||
      this.deps.getUserToolId() !== 'transform' ||
      this.deps.session.get() !== null
    ) {
      this.deps.pending.set(null);
    } else if (isTransformEligible(layer, document)) {
      this.deps.pending.set(null);
      this.begin(layerId);
    }
  }

  /** A genuine switch to another tool ends the wait; temporary holds never reach here. */
  onToolSwitch(to: ToolId): void {
    if (to !== 'transform') {
      if (this.processing) {
        this.cancel();
      }
      this.deps.pending.set(null);
    }
  }

  /** Escape ends a wait once no session is open; returns whether it did. */
  onEscape(): boolean {
    if (this.deps.pending.get() === null) {
      return false;
    }
    this.deps.pending.set(null);
    return true;
  }

  /** A replay lands on a document the request never saw, so the wait ends. */
  onReplay(): void {
    if (this.processing) {
      this.cancel();
    }
    this.deps.pending.set(null);
  }

  /** A replaced document drops the session and the wait without restoring anything. */
  onDocumentReplaced(): void {
    this.cancel();
    this.deps.pending.set(null);
  }

  /** Pixels that could not be persisted, with no retry to follow, will not open a session; say so. */
  onUploadFailed(layerId: string): void {
    if (this.deps.pending.get() === layerId) {
      this.deps.pending.set(null);
      this.deps.reportRefusal('not-ready');
    }
  }

  update(transform: LayerTransform): void {
    const session = this.deps.session.get();
    if (this.disposed || !session || this.processing) {
      return;
    }
    this.deps.session.set({ ...session, transform, applyFailed: false });
    this.deps.setOverride(session.layerId, transform);
    this.deps.invalidate({ layers: [session.layerId], overlay: true });
  }

  cancel(): void {
    this.processing?.abort.abort();
    this.processing = null;
    const session = this.deps.session.get();
    if (!session) {
      return;
    }
    this.deps.setOverride(session.layerId, null);
    this.deps.session.set(null);
    this.deps.invalidate({ layers: [session.layerId], overlay: true });
  }

  /** A step that makes `change` and installs matching whole-layer pixels, or rolls the change back. */
  private bakeStep(layerId: string, change: BakeChange, rect: Rect, pixels: ImageData): EditStep {
    const surface = this.deps.backend.createSurface(rect.width, rect.height);
    surface.ctx.putImageData(pixels, 0, 0);
    const prepared = this.deps.ctx.preparePixels(layerId, rect, surface);
    const install = (): void => this.deps.ctx.installPrepared(prepared);
    if (change.kind === 'contract') {
      return replaceLayerStep(this.deps.ctx, change.to, prepared, { restore: change.from });
    }
    const hasTransform =
      (transform: LayerTransform) =>
      (document: CanvasDocumentContractV3 | null): boolean => {
        const current = getDocumentLayer(document, layerId);
        return !!current && unchanged(current.transform, transform);
      };
    return {
      accepted: hasTransform(change.to),
      install,
      mutation: { id: layerId, patch: { transform: change.to }, type: 'updateCanvasLayer' },
      rollback: {
        mutation: { id: layerId, patch: { transform: change.from }, type: 'updateCanvasLayer' },
        restored: hasTransform(change.from),
      },
    };
  }

  /** Commits the session; a refusal keeps it open and is reported, any other failure ends it. */
  apply(): void {
    if (this.disposed || this.processing || !this.deps.canEdit() || this.deps.isGestureActive()) {
      return;
    }
    const session = this.deps.session.get();
    if (!session) {
      return;
    }
    const document = this.deps.getDocument();
    const layer = getDocumentLayer(document, session.layerId);
    const size = document && layer ? hittableLayerSize(layer, document) : null;
    if (
      !document ||
      !layer ||
      !size ||
      !isRenderableLayer(layer) ||
      !isLeafEditable(lookupDocumentLeaf(document, session.layerId)) ||
      unchanged(session.transform, session.startTransform)
    ) {
      this.cancel();
      return;
    }
    const smoothing = this.deps.getOptions?.();
    if (
      smoothing?.smoothing &&
      this.deps.resample &&
      (layer.type === 'raster' || layer.type === 'control') &&
      (layer.source.type === 'paint' || layer.source.type === 'image')
    ) {
      if (!this.processing) {
        void this.applyRasterized(session, layer, smoothing.resampleMode);
      }
      return;
    }
    if (
      layer.type === 'raster' &&
      layer.source.type === 'paint' &&
      (isPositionDependentAdjustments(layer.adjustments) || hasSpatialAdjustments(layer.adjustments))
    ) {
      void this.applyRasterized(session, layer, null);
      return;
    }
    const source = layer.type === 'raster' || layer.type === 'control' ? layer.source : null;
    if (
      source?.type === 'image' ||
      source?.type === 'shape' ||
      source?.type === 'gradient' ||
      source?.type === 'text'
    ) {
      const result = this.deps.commitStructural(
        'Transform layer',
        { id: session.layerId, patch: { transform: session.transform }, type: 'updateCanvasLayer' },
        { id: session.layerId, patch: { transform: session.startTransform }, type: 'updateCanvasLayer' }
      );
      this.settle(result.status);
      return;
    }
    const cache = this.deps.getCache(layer.id);
    if (source?.type !== 'paint' || !cache || isEmpty(cache.rect)) {
      this.cancel();
      return;
    }
    const beforeRect = { ...cache.rect };
    const matrix = bakeMatrix(session.transform);
    const afterRect = roundOut(transformBounds(matrix, beforeRect));
    const beforeBytes = rgbaBytes(beforeRect);
    const afterBytes = rgbaBytes(afterRect);
    const txn = this.deps.ctx.begin({ historyBytes: beforeBytes + afterBytes + HISTORY_ENTRY_OVERHEAD_BYTES });
    if (!('publish' in txn)) {
      this.settle(txn.status);
      return;
    }
    try {
      // The bake surface lives beside the cache until the step installs its pixels.
      if (!txn.reserveRaster(afterBytes * 3)) {
        this.settle('over-budget');
        return;
      }
      const before = cache.surface.ctx.getImageData(0, 0, beforeRect.width, beforeRect.height);
      const baked = this.deps.backend.createSurface(afterRect.width, afterRect.height);
      const context = baked.ctx;
      context.setTransform(1, 0, 0, 1, 0, 0);
      context.clearRect(0, 0, afterRect.width, afterRect.height);
      context.imageSmoothingEnabled = true;
      context.setTransform(matrix.a, matrix.b, matrix.c, matrix.d, matrix.e - afterRect.x, matrix.f - afterRect.y);
      context.drawImage(cache.surface.canvas, beforeRect.x, beforeRect.y);
      context.setTransform(1, 0, 0, 1, 0, 0);
      const after = context.getImageData(0, 0, afterRect.width, afterRect.height);
      // A kept control original moves with the baked pixels (detaching when its frame would shear), in the same step.
      const rebased =
        layer.type === 'control' && layer.original !== undefined
          ? rebaseControlOriginal({ ...layer, transform: { ...IDENTITY } }, session.transform)
          : null;
      const change: BakeChange = rebased
        ? { from: layer, kind: 'contract', to: rebased }
        : { from: { ...session.startTransform }, kind: 'transform', to: IDENTITY };
      const forward = () => this.bakeStep(layer.id, change, afterRect, after);
      const backward = () => this.bakeStep(layer.id, reversed(change), beforeRect, before);
      const result = txn.publish('Transform layer', forward(), {
        bytes: before.data.byteLength + after.data.byteLength + HISTORY_ENTRY_OVERHEAD_BYTES,
        // A contract bake that detaches drops the original from the document; undo restores it, so the entry holds it.
        heldAssetRefs:
          change.kind === 'contract' ? collectHistoryMediaRefs(change.from, change.to) : NO_HELD_ASSET_REFS,
        redo: () => this.replay(forward, afterRect),
        undo: () => this.replay(backward, beforeRect),
      });
      if (result.status === 'committed') {
        this.deps.rebaseLineEnd(layer.id, session.transform);
      }
      this.settle(result.status);
    } catch (error) {
      if (this.deps.reportFailure) {
        this.deps.reportFailure(layer.id, error);
      } else {
        this.deps.reportRefusal('not-ready');
      }
    } finally {
      txn.end();
    }
  }

  private markApplyFailed(): void {
    const session = this.deps.session.get();
    if (session) {
      this.deps.session.set({ ...session, applyFailed: true });
    }
  }

  /** Snapshot before the network wait; no transaction or mutable cache is held across it. */
  private async applyRasterized(
    session: TransformSession,
    layer: CanvasLayerContract,
    mode: TransformOptions['resampleMode'] | null
  ): Promise<void> {
    const cache = this.deps.getCache(layer.id);
    if (
      !cache ||
      cache.stale ||
      !cache.hasPublishedPixels ||
      isEmpty(cache.rect) ||
      (mode !== null && !this.deps.resample)
    ) {
      this.markApplyFailed();
      this.deps.reportRefusal('not-ready');
      return;
    }
    const beforeRect = { ...cache.rect };
    const matrix = bakeMatrix(session.transform);
    const afterRect = roundOut(transformBounds(matrix, beforeRect));
    const bakesAdjustments =
      layer.type === 'raster' &&
      (isPositionDependentAdjustments(layer.adjustments) || hasSpatialAdjustments(layer.adjustments));
    // Undo, adjustment input/source, transformed pixels, decode and prepared cache buffers.
    const lease = this.deps.ctx.reserveRaster(
      rgbaBytes(beforeRect) * (bakesAdjustments ? 3 : 1) + rgbaBytes(afterRect) * (mode === null ? 4 : 8)
    );
    if (!lease) {
      this.markApplyFailed();
      this.deps.reportRefusal('over-budget');
      return;
    }
    const version = cache.version;
    const pending = { abort: new AbortController(), layer };
    this.processing = pending;
    this.deps.session.set({ ...session, processing: true, processingPhase: 'preparing', applyFailed: false });
    const current = () =>
      !this.disposed &&
      this.processing === pending &&
      !pending.abort.signal.aborted &&
      getDocumentLayer(this.deps.getDocument(), layer.id) === layer &&
      this.deps.getDocument()?.selectedLayerId === layer.id &&
      this.deps.session.get()?.layerId === layer.id &&
      this.deps.canEdit() &&
      isLeafEditable(lookupDocumentLeaf(this.deps.getDocument(), layer.id)) &&
      this.deps.getCache(layer.id)?.version === version;
    try {
      const before = cache.surface.ctx.getImageData(0, 0, beforeRect.width, beforeRect.height);
      let source = cache.surface;
      if (bakesAdjustments && layer.type === 'raster') {
        // A worker may transfer this buffer; the independent raw snapshot remains available to undo.
        const input = cache.surface.ctx.getImageData(0, 0, beforeRect.width, beforeRect.height);
        const adjusted = await adjustExactly(input, layer.adjustments, beforeRect, {
          run: this.deps.runSpatial,
          signal: pending.abort.signal,
        });
        if (!current()) {
          return;
        }
        if (adjusted.status !== 'ok') {
          if (adjusted.status !== 'aborted') {
            this.markApplyFailed();
            this.deps.reportRefusal(adjustmentRefusal(this.deps.getDocument(), layer.id, adjusted.status));
          }
          return;
        }
        source = this.deps.backend.createSurface(beforeRect.width, beforeRect.height);
        source.ctx.putImageData(
          adjusted.data === input.data ? input : imageDataOf(adjusted.data, beforeRect.width, beforeRect.height),
          0,
          0
        );
      }
      const ratio = mode === null ? 1 : 2;
      const surface = this.deps.backend.createSurface(afterRect.width * ratio, afterRect.height * ratio);
      surface.ctx.imageSmoothingEnabled = true;
      surface.ctx.setTransform(
        matrix.a * ratio,
        matrix.b * ratio,
        matrix.c * ratio,
        matrix.d * ratio,
        (matrix.e - afterRect.x) * ratio,
        (matrix.f - afterRect.y) * ratio
      );
      surface.ctx.drawImage(source.canvas, beforeRect.x, beforeRect.y);
      const after =
        mode === null
          ? surface.ctx.getImageData(0, 0, afterRect.width, afterRect.height)
          : await this.deps.resample!(surface, afterRect, mode, pending.abort.signal, (phase) => {
              if (current()) {
                const active = this.deps.session.get();
                if (active) {
                  this.deps.session.set({ ...active, processingPhase: phase });
                }
              }
            });
      if (!current()) {
        return;
      }
      if (after.width !== afterRect.width || after.height !== afterRect.height) {
        throw new Error('Transform resampling returned unexpected dimensions');
      }
      const txn = this.deps.ctx.begin({
        historyBytes: before.data.byteLength + after.data.byteLength + HISTORY_ENTRY_OVERHEAD_BYTES,
      });
      if (!('publish' in txn)) {
        this.markApplyFailed();
        this.deps.reportRefusal(txn.status);
        return;
      }
      try {
        let paint = {
          ...layer,
          source: { bitmap: null, offset: { x: afterRect.x, y: afterRect.y }, type: 'paint' as const },
          transform: { ...IDENTITY },
        } as CanvasLayerContract;
        if (paint.type === 'raster' && bakesAdjustments) {
          delete paint.adjustments;
        }
        if (paint.type === 'control' && paint.original !== undefined) {
          paint = rebaseControlOriginal(paint, session.transform);
        }
        const change: BakeChange = { from: layer, kind: 'contract', to: paint };
        const forward = () => this.bakeStep(layer.id, change, afterRect, after);
        const backward = () => this.bakeStep(layer.id, reversed(change), beforeRect, before);
        const preparedStep = forward();
        let result: EditPublishResult;
        // The mirror notifies synchronously for both our replacement and its possible rollback.
        // Only the layer-identity check waits for publication's verified outcome; selection/locks still cancel.
        this.publishingResampled = true;
        try {
          result = txn.publish('Transform layer', preparedStep, {
            bytes: before.data.byteLength + after.data.byteLength + HISTORY_ENTRY_OVERHEAD_BYTES,
            heldAssetRefs: collectHistoryMediaRefs(layer, paint),
            redo: () => this.replay(forward, afterRect),
            undo: () => this.replay(backward, beforeRect),
          });
        } finally {
          this.publishingResampled = false;
        }
        if (result.status === 'committed') {
          this.deps.rebaseLineEnd(layer.id, session.transform);
        }
        this.settleResampled(result);
      } finally {
        txn.end();
      }
    } catch (error) {
      if (current()) {
        this.markApplyFailed();
        if (this.deps.reportFailure) {
          this.deps.reportFailure(layer.id, error);
        } else {
          this.deps.reportRefusal('not-ready');
        }
      }
    } finally {
      lease.release();
      if (this.processing === pending) {
        this.processing = null;
        const active = this.deps.session.get();
        if (active) {
          this.deps.session.set({ ...active, processing: false, processingPhase: undefined });
        }
      }
    }
  }

  /** Replays a bake step, reserving the whole-layer pixels it installs; throws, leaving it in place, when they do not fit. */
  private replay(step: () => EditStep, rect: Rect): void {
    withReplayReservation(this.deps.ctx, rgbaBytes(rect) * 2, () => this.deps.ctx.applyStep(step()));
  }

  private settleResampled(result: EditPublishResult): void {
    if (result.status === 'committed') {
      this.cancel();
      return;
    }
    if (result.status === 'postcondition-failed' && result.recovered !== 'reverted') {
      // The mutation context reports unsafe rollback failures; its document can no longer support this preview.
      this.cancel();
      return;
    }
    this.markApplyFailed();
    this.deps.reportRefusal(isRefusal(result.status) ? result.status : 'not-ready');
  }

  private settle(status: StructuralCommitResult['status']): void {
    if (isRefusal(status)) {
      this.deps.reportRefusal(status);
      return;
    }
    this.cancel();
  }

  dispose(): void {
    if (this.disposed) {
      return;
    }
    this.cancel();
    this.deps.pending.set(null);
    this.disposed = true;
  }
}
