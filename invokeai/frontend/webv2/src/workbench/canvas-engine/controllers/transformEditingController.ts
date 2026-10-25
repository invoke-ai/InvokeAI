import type { CanvasEditRefusal, StructuralCommitResult } from '@workbench/canvas-engine/capabilities';
import type { CanvasDocumentContractV3, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { TransformSession } from '@workbench/canvas-engine/engineStores';
import type { CanvasProjectMutation } from '@workbench/canvas-engine/mutationContracts';
import type { LayerCacheEntry } from '@workbench/canvas-engine/render/layerCache';
import type { RasterBackend } from '@workbench/canvas-engine/render/raster';
import type { LayerTransform } from '@workbench/canvas-engine/transform/transformMath';
import type { Rect } from '@workbench/canvas-engine/types';

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
import { hittableLayerSize } from '@workbench/canvas-engine/tools/moveHitTest';
import { bakeMatrix } from '@workbench/canvas-engine/transform/transformMath';

import type { CanvasMutationContext, EditStep } from './mutationContext';

import { replaceLayerStep, rgbaBytes, withReplayReservation } from './editSteps';

export interface TransformEditingControllerOptions {
  readonly session: { get(): TransformSession | null; set(value: TransformSession | null): void };
  readonly backend: RasterBackend;
  readonly ctx: Pick<CanvasMutationContext, 'applyStep' | 'begin' | 'installPrepared' | 'reserveRaster'>;
  readonly getDocument: () => CanvasDocumentContractV3 | null;
  readonly getCache: (layerId: string) => LayerCacheEntry | null;
  readonly setOverride: (layerId: string, transform: LayerTransform | null) => void;
  /** Installs whole-layer pixels at a layer-local rect, marking them painted and dirty. */
  readonly restoreCache: (layerId: string, rect: Rect, pixels: ImageData) => void;
  readonly commitStructural: (
    label: string,
    forward: CanvasProjectMutation,
    inverse: CanvasProjectMutation
  ) => StructuralCommitResult;
  readonly reportRefusal: (refusal: CanvasEditRefusal) => void;
  readonly canEdit: () => boolean;
  readonly isGestureActive: () => boolean;
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

/** Owns transform session state, previews, param commits, and paint bakes. */
export class TransformEditingController {
  private disposed = false;

  constructor(private readonly deps: TransformEditingControllerOptions) {}

  private clearOverride(): void {
    const session = this.deps.session.get();
    if (session) {
      this.deps.setOverride(session.layerId, null);
    }
  }

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
    this.clearOverride();
    const start = { ...layer.transform };
    this.deps.session.set({ layerId, startTransform: start, transform: start });
    this.deps.setOverride(layerId, start);
    this.deps.invalidate({ layers: [layerId], overlay: true });
  }

  update(transform: LayerTransform): void {
    const session = this.deps.session.get();
    if (this.disposed || !session) {
      return;
    }
    this.deps.session.set({ ...session, transform });
    this.deps.setOverride(session.layerId, transform);
    this.deps.invalidate({ layers: [session.layerId], overlay: true });
  }

  cancel(): void {
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
    const install = (): void => this.deps.restoreCache(layerId, rect, pixels);
    if (change.kind === 'contract') {
      return { ...replaceLayerStep(this.deps.ctx, change.to, null, { restore: change.from }), install };
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
    if (this.disposed || !this.deps.canEdit() || this.deps.isGestureActive()) {
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
      if (!txn.reserveRaster(afterBytes)) {
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
      const forward = this.bakeStep(layer.id, change, afterRect, after);
      const backward = this.bakeStep(layer.id, reversed(change), beforeRect, before);
      const result = txn.publish('Transform layer', forward, {
        bytes: before.data.byteLength + after.data.byteLength + HISTORY_ENTRY_OVERHEAD_BYTES,
        // A contract bake that detaches drops the original from the document; undo restores it, so the entry holds it.
        heldAssetRefs:
          change.kind === 'contract' ? collectHistoryMediaRefs(change.from, change.to) : NO_HELD_ASSET_REFS,
        redo: () => this.replay(forward, afterRect),
        undo: () => this.replay(backward, beforeRect),
      });
      this.settle(result.status);
    } finally {
      txn.end();
    }
  }

  /** Replays a bake step, reserving the whole-layer pixels it installs; throws, leaving it in place, when they do not fit. */
  private replay(step: EditStep, rect: Rect): void {
    withReplayReservation(this.deps.ctx, rgbaBytes(rect), () => this.deps.ctx.applyStep(step));
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
    this.disposed = true;
  }
}
