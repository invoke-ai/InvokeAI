/** Transactional pixel editing for controls and destructively edited raster images. */

import type { CanvasControlLayerContract, CanvasDocumentContractV2 } from '@workbench/canvas-engine/contracts';
import type { BitmapStore } from '@workbench/canvas-engine/document/bitmapStore';
import type { History } from '@workbench/canvas-engine/history/history';
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

import { getSourceContentRect } from '@workbench/canvas-engine/document/sources';
import {
  bakePixelEditSurface,
  buildMaterializedPixelLayer,
  decidePixelEdit,
  type PixelEditableLayer,
} from '@workbench/canvas-engine/editing/controlPixelEdit';
import { createImagePatchEntry } from '@workbench/canvas-engine/history/imagePatch';
import { createLayerSnapshotEntry } from '@workbench/canvas-engine/history/layerSnapshot';
import { isEmpty } from '@workbench/canvas-engine/math/rect';

export interface PixelEditControllerOptions {
  readonly applyImagePatch: ImagePatchApply;
  readonly backend: RasterBackend;
  readonly bitmapStore: Pick<BitmapStore, 'discardLayer' | 'markLayerDirty' | 'suspendLayer'>;
  readonly canEdit: () => boolean;
  readonly deleteDerived: (layerId: string) => void;
  readonly dispatchReplacement: (layer: PixelEditableLayer) => void;
  readonly endBurst: () => void;
  readonly getActiveProjectId: () => string | null;
  readonly getAdjustedSurface: (layer: PixelEditableLayer, entry: LayerCacheEntry) => RasterSurface | null;
  readonly getDocument: () => CanvasDocumentContractV2 | null;
  readonly getTransformSession: () => unknown;
  readonly history: History;
  readonly installPrepared: (prepared: PreparedLayerCacheReplacement, persist?: boolean) => void;
  readonly invalidate: (layerId: string, overlay?: boolean) => void;
  readonly isCacheReady: (layer: PixelEditableLayer, document: CanvasDocumentContractV2) => boolean;
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

/** Owns the exclusive direct/materialized pixel transaction for control layers and raster images. */
export class PixelEditController {
  private open: { cancel: () => void; layerId: string } | null = null;

  constructor(private readonly options: PixelEditControllerOptions) {}

  cancel(): void {
    this.open?.cancel();
  }

  isOpenFor(layerIds: readonly string[]): boolean {
    return this.open !== null && layerIds.includes(this.open.layerId);
  }

  private applySnapshot: LayerPixelSnapshotApply = (snapshot) => {
    const pixels = this.options.backend.createSurface(snapshot.rect.width, snapshot.rect.height);
    if (snapshot.pixels) {
      pixels.ctx.putImageData(snapshot.pixels, 0, 0);
    }
    const prepared = this.options.preparePixels(snapshot.layer.id, snapshot.rect, pixels);
    this.options.dispatchReplacement(snapshot.layer);
    try {
      this.options.bitmapStore.discardLayer(snapshot.layer.id);
    } catch {
      // Persistence bookkeeping is ancillary after reducer acceptance.
    }
    this.options.installPrepared(prepared, snapshot.layer.source.type === 'paint');
  };

  begin(layerId: string): PixelEditTransaction | null {
    const o = this.options;
    const document = o.getDocument();
    const layer = document?.layers.find((candidate) => candidate.id === layerId);
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
    const contentRect = getSourceContentRect(layer, document);
    const decision = decidePixelEdit({
      hasSourceContent: !isEmpty(contentRect),
      isCacheReady: o.isCacheReady(layer, document),
      layer,
    });
    if (decision.status === 'rejected') {
      return null;
    }
    if (decision.status === 'direct') {
      if (layer.type !== 'control') {
        return null;
      }
      return this.beginDirect(layerId, layer);
    }
    return this.beginMaterialized(layerId, layer, contentRect);
  }

  private beginDirect(layerId: string, layer: CanvasControlLayerContract): PixelEditTransaction | null {
    const o = this.options;
    const originalEntry = o.layers.get(layerId);
    let originalPixels: ImageData | null = null;
    if (originalEntry && !isEmpty(originalEntry.rect)) {
      try {
        originalPixels = originalEntry.surface.ctx.getImageData(
          0,
          0,
          originalEntry.rect.width,
          originalEntry.rect.height
        );
      } catch {
        return null;
      }
    }
    const original = originalEntry
      ? {
          hasPublishedPixels: originalEntry.hasPublishedPixels,
          lastUsed: originalEntry.lastUsed,
          pixels: originalPixels,
          rect: { ...originalEntry.rect },
          stale: originalEntry.stale,
          surface: originalEntry.surface,
          version: originalEntry.version,
        }
      : null;
    const releasePersistence = o.bitmapStore.suspendLayer(layerId);
    let closed = false;
    let owner: { cancel: () => void; layerId: string };
    const close = (): boolean => {
      if (closed || this.open !== owner) {
        return false;
      }
      closed = true;
      this.open = null;
      return true;
    };
    const restore = (): void => {
      try {
        if (!original) {
          o.layers.delete(layerId);
        } else {
          original.surface.resize(original.rect.width, original.rect.height);
          if (original.pixels) {
            original.surface.ctx.putImageData(original.pixels, 0, 0);
          }
          const current = o.layers.get(layerId) ?? o.layers.getOrCreateRect(layerId, original.rect);
          Object.assign(current, {
            hasPublishedPixels: original.hasPublishedPixels,
            lastUsed: original.lastUsed,
            rect: { ...original.rect },
            stale: original.stale,
            surface: original.surface,
            version: original.version,
          });
        }
      } finally {
        o.deleteDerived(layerId);
        o.invalidate(layerId);
      }
    };
    const restoreAndRelease = (): void => {
      try {
        restore();
      } finally {
        releasePersistence();
      }
    };
    const cancel = (): void => {
      if (close()) {
        restoreAndRelease();
      }
    };
    const commitPatch = (label: string, patch: PixelEditPatch): boolean => {
      if (closed || this.open !== owner) {
        return false;
      }
      if (isImageDataEqual(patch.before, patch.after)) {
        cancel();
        return false;
      }
      const document = o.getDocument();
      if (
        o.getActiveProjectId() !== o.projectId ||
        !o.canEdit() ||
        !o.isOperationIdle() ||
        document?.selectedLayerId !== layerId ||
        document.layers.find((candidate) => candidate.id === layerId) !== layer
      ) {
        cancel();
        return false;
      }
      close();
      let entry = null;
      try {
        if (!o.history.isApplying()) {
          entry = createImagePatchEntry({
            after: patch.after,
            apply: o.applyImagePatch,
            before: patch.before,
            label,
            layerId,
            rect: patch.rect,
          });
        }
      } catch (error) {
        restoreAndRelease();
        throw error;
      }
      try {
        // Once the pixels are accepted, persistence and history are the
        // authoritative consequences. Publish UI/render notifications only
        // afterwards: a faulty observer must not make a visible edit ephemeral.
        if (entry) {
          o.history.push(entry);
        }
        o.bitmapStore.markLayerDirty(layerId);
        o.endBurst();
        try {
          o.notifyPainted(layerId);
        } catch {
          // The live cache already owns the accepted pixels and is dirty. A
          // later render invalidation reconciles ancillary UI observers.
        }
      } finally {
        releasePersistence();
      }
      return true;
    };
    const transaction: PixelEditTransaction = {
      cancel,
      commitPatch: (label, patch) => void commitPatch(label, patch),
      commitStroke: (event) => {
        if (event.layerId !== layerId) {
          cancel();
          return;
        }
        if (
          commitPatch(event.tool === 'eraser' ? 'Eraser stroke' : 'Brush stroke', {
            after: event.afterImageData,
            before: event.beforeImageData,
            rect: event.dirtyRect,
          })
        ) {
          o.publishStroke(event);
        }
      },
      layerId,
    };
    owner = { cancel, layerId };
    this.open = owner;
    return transaction;
  }

  private beginMaterialized(
    layerId: string,
    layer: PixelEditableLayer,
    contentRect: Rect
  ): PixelEditTransaction | null {
    const o = this.options;
    const originalEntry = o.layers.get(layerId);
    const original = originalEntry
      ? {
          hasPublishedPixels: originalEntry.hasPublishedPixels,
          lastUsed: originalEntry.lastUsed,
          rect: { ...originalEntry.rect },
          stale: originalEntry.stale,
          surface: originalEntry.surface,
          version: originalEntry.version,
        }
      : null;
    const beforeRect = original?.rect ?? { ...contentRect };
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
    const restore = (): void => {
      try {
        if (original) {
          const current = o.layers.get(layerId);
          if (current) {
            Object.assign(current, { ...original, rect: { ...original.rect } });
          }
        } else {
          o.layers.delete(layerId);
        }
      } finally {
        o.deleteDerived(layerId);
        o.setTransformOverride(layerId, null);
        o.invalidate(layerId, true);
      }
    };
    const releasePersistence = o.bitmapStore.suspendLayer(layerId);
    const restoreAndRelease = (): void => {
      try {
        restore();
      } finally {
        releasePersistence();
      }
    };
    try {
      const preview = originalEntry ?? o.layers.getOrCreateRect(layerId, prepared.rect);
      Object.assign(preview, {
        surface: prepared.surface,
        rect: { ...prepared.rect },
        hasPublishedPixels: true,
        stale: false,
      });
      preview.version += 1;
      o.deleteDerived(layerId);
      o.setTransformOverride(layerId, { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 });
      o.invalidate(layerId, true);
    } catch {
      restoreAndRelease();
      return null;
    }
    let closed = false;
    let owner: { cancel: () => void; layerId: string };
    const close = (): boolean => {
      if (closed || this.open !== owner) {
        return false;
      }
      closed = true;
      this.open = null;
      return true;
    };
    const cancel = (): void => {
      if (close()) {
        restoreAndRelease();
      }
    };
    const commit = (label: string, event?: StrokeCommittedEvent): void => {
      if (!close()) {
        return;
      }
      const document = o.getDocument();
      const edited = o.layers.get(layerId);
      if (
        o.getActiveProjectId() !== o.projectId ||
        !o.canEdit() ||
        !o.isOperationIdle() ||
        document?.selectedLayerId !== layerId ||
        document.layers.find((candidate) => candidate.id === layerId) !== layer ||
        !edited ||
        (event && event.layerId !== layerId)
      ) {
        restoreAndRelease();
        return;
      }
      let entry = null;
      try {
        const pixels = isEmpty(edited.rect)
          ? null
          : edited.surface.ctx.getImageData(0, 0, edited.rect.width, edited.rect.height);
        const materialized = buildMaterializedPixelLayer(layer, edited.rect);
        const after: LayerPixelSnapshot = { layer: materialized, pixels, rect: { ...edited.rect } };
        if (!o.history.isApplying()) {
          entry = createLayerSnapshotEntry({ after, apply: this.applySnapshot, before, label });
        }
        o.dispatchReplacement(materialized);
      } catch (error) {
        restoreAndRelease();
        throw error;
      }
      try {
        // The replacement has passed reducer and mirror postconditions. From
        // here on it must never be treated as rollback-safe, even if cleanup or
        // an observer fails.
        if (entry) {
          o.history.push(entry);
        }
        o.bitmapStore.markLayerDirty(layerId);
        o.setTransformOverride(layerId, null);
        o.endBurst();
        try {
          o.notifyPainted(layerId);
        } catch {
          // See the direct path: persistence is already authoritative.
        }
      } finally {
        releasePersistence();
      }
      if (event) {
        o.publishStroke(event);
      }
    };
    const transaction: PixelEditTransaction = {
      cancel,
      commitPatch: (label, patch) => (isImageDataEqual(patch.before, patch.after) ? cancel() : commit(label)),
      commitStroke: (event) =>
        isImageDataEqual(event.beforeImageData, event.afterImageData)
          ? cancel()
          : commit(event.tool === 'eraser' ? 'Eraser stroke' : 'Brush stroke', event),
      layerId,
    };
    owner = { cancel, layerId };
    this.open = owner;
    return transaction;
  }

  dispose(): void {
    this.cancel();
  }
}
