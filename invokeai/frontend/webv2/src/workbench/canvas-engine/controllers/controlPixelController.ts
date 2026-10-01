/** Transactional pixel editing for controls and destructively edited raster images. */

import type { CanvasEditRefusal } from '@workbench/canvas-engine/capabilities';
import type {
  CanvasControlLayerContract,
  CanvasDocumentContractV3,
  CanvasLayerContract,
} from '@workbench/canvas-engine/contracts';
import type { BitmapStore } from '@workbench/canvas-engine/document/bitmapStore';
import type { ImagePatchApply } from '@workbench/canvas-engine/history/imagePatch';
import type { LayerPixelSnapshot, LayerPixelSnapshotApply } from '@workbench/canvas-engine/history/layerSnapshot';
import type {
  LayerCacheEntry,
  LayerCacheStore,
  PreparedLayerCacheReplacement,
} from '@workbench/canvas-engine/render/layerCache';
import type { RasterBackend, RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { PixelEditTransaction, PixelEditPatch, StrokeCommittedEvent } from '@workbench/canvas-engine/tools/tool';
import type { LayerTransform } from '@workbench/canvas-engine/transform/transformMath';
import type { Rect } from '@workbench/canvas-engine/types';

import { areJsonValuesStructurallyEqual } from '@platform/core/json';
import { lookupDocumentLeaf } from '@workbench/canvas-engine/document-model/documentModel';
import { getDocumentLayer } from '@workbench/canvas-engine/document/documentIndex';
import { getSourceContentRect } from '@workbench/canvas-engine/document/sources';
import {
  bakePixelEditSurface,
  buildMaterializedPixelLayer,
  decidePixelEdit,
  type PixelEditableLayer,
} from '@workbench/canvas-engine/editing/controlPixelEdit';
import { HISTORY_ENTRY_OVERHEAD_BYTES } from '@workbench/canvas-engine/history/history';
import { createImagePatchEntry } from '@workbench/canvas-engine/history/imagePatch';
import { createLayerSnapshotEntry } from '@workbench/canvas-engine/history/layerSnapshot';
import { isEmpty } from '@workbench/canvas-engine/math/rect';
import { strokeCommitLabel } from '@workbench/canvas-engine/strokeCommit';

import type { CanvasMutationContext, EditStep, EditTransaction } from './mutationContext';

export interface PixelEditControllerOptions {
  readonly ctx: Pick<CanvasMutationContext, 'applyStep' | 'begin'>;
  readonly applyImagePatch: ImagePatchApply;
  readonly backend: RasterBackend;
  readonly bitmapStore: Pick<BitmapStore, 'discardLayer' | 'markLayerDirty' | 'suspendLayer'>;
  readonly canEdit: () => boolean;
  readonly deleteDerived: (layerId: string) => void;
  readonly getActiveProjectId: () => string | null;
  readonly getAdjustedSurface: (layer: PixelEditableLayer, entry: LayerCacheEntry) => RasterSurface | null;
  readonly getDocument: () => CanvasDocumentContractV3 | null;
  readonly getTransformSession: () => unknown;
  readonly installPrepared: (prepared: PreparedLayerCacheReplacement, persist?: boolean) => void;
  readonly invalidate: (layerId: string, overlay?: boolean) => void;
  readonly isCacheReady: (layer: PixelEditableLayer, document: CanvasDocumentContractV3) => boolean;
  readonly isOperationIdle: () => boolean;
  readonly layers: LayerCacheStore;
  readonly notifyPainted: (layerId: string) => void;
  readonly preparePixels: (
    layerId: string,
    rect: Rect,
    pixels: ReturnType<RasterBackend['createSurface']>
  ) => PreparedLayerCacheReplacement;
  readonly projectId: string;
  readonly publishStroke: (event: StrokeCommittedEvent) => void;
  readonly reportRefusal: (refusal: CanvasEditRefusal) => void;
  readonly setTransformOverride: (layerId: string, transform: LayerTransform | null) => void;
}

const isImageDataEqual = (left: ImageData, right: ImageData): boolean => {
  if (left.width !== right.width || left.height !== right.height || left.data.length !== right.data.length) {
    return false;
  }
  for (let index = 0; index < left.data.length; index += 1) {
    if (left.data[index] !== right.data[index]) {
      return false;
    }
  }
  return true;
};

/** Conversion reducers clone contracts, so the replacement postcondition compares by value. */
const hasLayerContract = (document: CanvasDocumentContractV3 | null, expected: CanvasLayerContract): boolean => {
  const current = getDocumentLayer(document, expected.id);
  return current !== undefined && areJsonValuesStructurallyEqual(current, expected);
};

/**
 * Owns the exclusive direct/materialized pixel transaction for control layers and raster images. Each is admitted
 * before the first pixel changes. A commit that records nothing leaves the edit open: callers restore the pixels
 * they touched, then cancel, which releases the transaction and, for a materialized edit, reinstates the unbaked
 * cache last.
 */
export class PixelEditController {
  // `publishing` while the edit records its own step: the document change it makes is not an external one.
  private open: { cancel: () => void; layerId: string; publishing: boolean } | null = null;

  constructor(private readonly options: PixelEditControllerOptions) {}

  cancel(): void {
    this.open?.cancel();
  }

  isOpenFor(layerIds: readonly string[]): boolean {
    return this.open !== null && !this.open.publishing && layerIds.includes(this.open.layerId);
  }

  /** Replays a whole-layer snapshot; throws, leaving the layer unchanged, when the document refuses it. */
  private applySnapshot: LayerPixelSnapshotApply = (snapshot) => {
    const o = this.options;
    const pixels = o.backend.createSurface(snapshot.rect.width, snapshot.rect.height);
    if (snapshot.pixels) {
      pixels.ctx.putImageData(snapshot.pixels, 0, 0);
    }
    const prepared = o.preparePixels(snapshot.layer.id, snapshot.rect, pixels);
    o.ctx.applyStep({
      accepted: (document) => hasLayerContract(document, snapshot.layer),
      install: () => o.installPrepared(prepared, snapshot.layer.source.type === 'paint'),
      mutation: { layer: snapshot.layer, layerId: snapshot.layer.id, type: 'replaceCanvasLayer' },
      notify: () => o.bitmapStore.discardLayer(snapshot.layer.id),
    });
  };

  /** `gesture` admits an edit the gesture in progress starts (a stroke's first press). */
  begin(layerId: string, options: { gesture?: boolean } = {}): PixelEditTransaction | null {
    const o = this.options;
    const document = o.getDocument();
    const layer = getDocumentLayer(document, layerId);
    if (
      !o.canEdit() ||
      !document ||
      document.selectedLayerId !== layerId ||
      !layer ||
      (layer.type !== 'control' && !(layer.type === 'raster' && layer.source.type === 'image')) ||
      this.open ||
      !o.isOperationIdle() ||
      o.getTransformSession()
    ) {
      return null;
    }
    const leaf = lookupDocumentLeaf(document, layerId);
    const contentRect = getSourceContentRect(layer, document);
    const decision = decidePixelEdit({
      contributionEnabled: leaf?.contributionEnabled ?? false,
      effectiveLocked: leaf?.effectiveLocked ?? true,
      hasSourceContent: !isEmpty(contentRect),
      isCacheReady: o.isCacheReady(layer, document),
      layer,
    });
    if (decision.status === 'rejected' || (decision.status === 'direct' && layer.type !== 'control')) {
      return null;
    }
    const entry = o.layers.get(layerId);
    const beforeBytes = entry && !isEmpty(entry.rect) ? entry.rect.width * entry.rect.height * 4 : 0;
    // A materialized edit records the whole layer before and after.
    const historyBytes =
      decision.status === 'direct' ? HISTORY_ENTRY_OVERHEAD_BYTES : beforeBytes * 2 + HISTORY_ENTRY_OVERHEAD_BYTES;
    const txn = o.ctx.begin({ gesture: options.gesture, historyBytes });
    if (!('publish' in txn)) {
      o.reportRefusal(txn.status);
      return null;
    }
    // The unbaked surface stays alive outside the cache's accounting until the edit ends.
    if (decision.status !== 'direct' && !txn.reserveRaster(beforeBytes)) {
      txn.end();
      o.reportRefusal('over-budget');
      return null;
    }
    const opened =
      decision.status === 'direct'
        ? this.beginDirect(txn, layerId, layer as CanvasControlLayerContract)
        : this.beginMaterialized(txn, layerId, layer, contentRect);
    if (!opened) {
      txn.end();
    }
    return opened;
  }

  /** The shared lifecycle: exclusive ownership, persistence suspension and refusal reporting. */
  private openTransaction(
    txn: EditTransaction,
    layerId: string,
    rollback: () => void
  ): {
    readonly owns: () => boolean;
    readonly close: (restore: boolean) => void;
    readonly grow: (bytes: number) => boolean;
    readonly refused: () => void;
    readonly publish: <T>(run: () => T) => T;
  } {
    const o = this.options;
    const releasePersistence = o.bitmapStore.suspendLayer(layerId);
    let closed = false;
    let refusalReported = false;
    const owner = {
      cancel: () => close(true),
      layerId,
      publishing: false,
    };
    const owns = (): boolean => !closed && this.open === owner;
    const close = (restore: boolean): void => {
      if (!owns()) {
        return;
      }
      closed = true;
      this.open = null;
      try {
        if (restore) {
          rollback();
        }
      } finally {
        try {
          releasePersistence();
        } finally {
          txn.end();
        }
      }
    };
    const refused = (): void => {
      if (!refusalReported) {
        refusalReported = true;
        o.reportRefusal('over-budget');
      }
    };
    this.open = owner;
    return {
      close,
      grow: (bytes) => {
        if (txn.growHistory(bytes)) {
          return true;
        }
        refused();
        return false;
      },
      owns,
      publish: (run) => {
        owner.publishing = true;
        try {
          return run();
        } finally {
          owner.publishing = false;
        }
      },
      refused,
    };
  }

  private isStillTarget(layerId: string, layer: PixelEditableLayer): boolean {
    const o = this.options;
    const document = o.getDocument();
    return (
      o.getActiveProjectId() === o.projectId &&
      o.canEdit() &&
      o.isOperationIdle() &&
      document?.selectedLayerId === layerId &&
      getDocumentLayer(document, layerId) === layer
    );
  }

  private beginDirect(txn: EditTransaction, layerId: string, layer: CanvasControlLayerContract): PixelEditTransaction {
    const o = this.options;
    const original = o.layers.captureState(layerId);
    const lifecycle = this.openTransaction(txn, layerId, () => {
      // The caller already put back the pixels and extent it touched; reinstate the exact entry, version included,
      // so guards captured before the edit stay current. An extent it could not restore falls back to rasterizing.
      const surface = original?.surface;
      if (!original || (surface!.width === original.rect.width && surface!.height === original.rect.height)) {
        o.layers.restoreState(layerId, original);
      } else {
        o.layers.invalidate(layerId);
      }
      o.deleteDerived(layerId);
      o.invalidate(layerId);
    });
    const commitPatch = (label: string, patch: PixelEditPatch, event?: StrokeCommittedEvent): boolean => {
      if (!lifecycle.owns()) {
        return false;
      }
      if (isImageDataEqual(patch.before, patch.after) || !this.isStillTarget(layerId, layer)) {
        return false;
      }
      const entry = createImagePatchEntry({
        after: patch.after,
        apply: o.applyImagePatch,
        before: patch.before,
        label,
        layerId,
        rect: patch.rect,
      });
      const result = lifecycle.publish(() =>
        txn.publish(
          label,
          {
            notify: () => {
              o.bitmapStore.markLayerDirty(layerId);
              o.notifyPainted(layerId);
              if (event) {
                o.publishStroke(event);
              }
            },
          },
          entry,
          { origin: 'system' }
        )
      );
      if (result.status === 'over-budget') {
        lifecycle.refused();
      }
      if (result.status === 'committed') {
        lifecycle.close(false);
      }
      return result.status === 'committed';
    };
    return {
      cancel: () => lifecycle.close(true),
      commit: (event) =>
        event.layerId === layerId
          ? commitPatch(
              strokeCommitLabel(event.tool),
              { after: event.afterImageData, before: event.beforeImageData, rect: event.dirtyRect },
              event
            )
          : false,
      commitPatch: (label, patch) => commitPatch(label, patch),
      grow: lifecycle.grow,
      layerId,
    };
  }

  private beginMaterialized(
    txn: EditTransaction,
    layerId: string,
    layer: PixelEditableLayer,
    contentRect: Rect
  ): PixelEditTransaction | null {
    const o = this.options;
    const originalEntry = o.layers.get(layerId);
    const original = o.layers.captureState(layerId);
    const beforeRect = original ? { ...original.rect } : { ...contentRect };
    let beforePixels: ImageData | null = null;
    if (!isEmpty(beforeRect)) {
      if (!originalEntry) {
        return null;
      }
      try {
        beforePixels = originalEntry.surface.ctx.getImageData(0, 0, beforeRect.width, beforeRect.height);
      } catch {
        return null;
      }
    }
    let prepared: PreparedLayerCacheReplacement;
    try {
      if (originalEntry && !isEmpty(originalEntry.rect)) {
        const adjusted = layer.type === 'raster' ? o.getAdjustedSurface(layer, originalEntry) : null;
        const baked = bakePixelEditSurface({
          backend: o.backend,
          // Raster presentation adjustments precede transforms in the normal
          // compositor. Bake from that adjusted surface so interpolation occurs
          // in the same order and untouched pixels remain visually identical.
          source: adjusted ?? originalEntry.surface,
          sourceRect: originalEntry.rect,
          transform: layer.transform,
        });
        prepared = o.preparePixels(layerId, baked.rect, baked.surface);
      } else {
        prepared = o.preparePixels(layerId, { height: 0, width: 0, x: 0, y: 0 }, o.backend.createSurface(0, 0));
      }
    } catch {
      return null;
    }
    const before: LayerPixelSnapshot = { layer: structuredClone(layer), pixels: beforePixels, rect: beforeRect };
    const lifecycle = this.openTransaction(txn, layerId, () => {
      try {
        o.layers.restoreState(layerId, original);
      } finally {
        o.deleteDerived(layerId);
        o.setTransformOverride(layerId, null);
        o.invalidate(layerId, true);
      }
    });
    try {
      o.layers.installReplacement(prepared);
      o.deleteDerived(layerId);
      o.setTransformOverride(layerId, { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 });
      o.invalidate(layerId, true);
    } catch {
      lifecycle.close(true);
      return null;
    }
    const commit = (label: string, event?: StrokeCommittedEvent): boolean => {
      if (!lifecycle.owns()) {
        return false;
      }
      const edited = o.layers.get(layerId);
      if (!this.isStillTarget(layerId, layer) || !edited || (event && event.layerId !== layerId)) {
        return false;
      }
      // A preparation failure throws with the edit still open; the caller restores its pixels and cancels.
      const pixels = isEmpty(edited.rect)
        ? null
        : edited.surface.ctx.getImageData(0, 0, edited.rect.width, edited.rect.height);
      const materialized = buildMaterializedPixelLayer(layer, edited.rect);
      const after: LayerPixelSnapshot = { layer: materialized, pixels, rect: { ...edited.rect } };
      const entry = createLayerSnapshotEntry({ after, apply: this.applySnapshot, before, label });

      const step: EditStep = {
        accepted: (document) => hasLayerContract(document, materialized),
        mutation: { layer: materialized, layerId, type: 'replaceCanvasLayer' },
        notify: () => {
          o.bitmapStore.markLayerDirty(layerId);
          o.setTransformOverride(layerId, null);
          o.notifyPainted(layerId);
          if (event) {
            o.publishStroke(event);
          }
        },
        rollback: {
          mutation: { layer, layerId, type: 'replaceCanvasLayer' },
          restored: (document) => hasLayerContract(document, layer),
        },
      };
      const result = lifecycle.publish(() => txn.publish(label, step, entry));
      if (result.status === 'over-budget') {
        lifecycle.refused();
      }
      if (result.status === 'committed') {
        lifecycle.close(false);
      }
      return result.status === 'committed';
    };
    return {
      cancel: () => lifecycle.close(true),
      commit: (event) =>
        isImageDataEqual(event.beforeImageData, event.afterImageData)
          ? false
          : commit(strokeCommitLabel(event.tool), event),
      commitPatch: (label, patch) => (isImageDataEqual(patch.before, patch.after) ? false : commit(label)),
      // The entry is two whole-layer snapshots, admitted at begin; the stroke's own footprint is never retained.
      // A cache the stroke grew past that admission grows it at publication instead.
      grow: () => lifecycle.owns(),
      layerId,
    };
  }

  dispose(): void {
    this.cancel();
  }
}
