import type { TransformSession } from '@workbench/canvas-engine/engineStores';
import type { HistoryEntry } from '@workbench/canvas-engine/history/history';
import type { LayerCacheEntry } from '@workbench/canvas-engine/render/layerCache';
import type { RasterBackend, RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { LayerTransform } from '@workbench/canvas-engine/transform/transformMath';
import type { Rect } from '@workbench/canvas-engine/types';
import type { CanvasDocumentContractV2 } from '@workbench/types';
import type { WorkbenchAction } from '@workbench/workbenchState';

import { isRenderableLayer } from '@workbench/canvas-engine/document/sources';
import { createDocumentPatchEntry } from '@workbench/canvas-engine/history/documentPatch';
import { isEmpty, roundOut, transformBounds } from '@workbench/canvas-engine/math/rect';
import { hittableLayerSize } from '@workbench/canvas-engine/tools/moveHitTest';
import { bakeMatrix } from '@workbench/canvas-engine/transform/transformMath';

export interface TransformEditingControllerOptions {
  readonly session: { get(): TransformSession | null; set(value: TransformSession | null): void };
  readonly backend: RasterBackend;
  readonly getDocument: () => CanvasDocumentContractV2 | null;
  readonly getCache: (layerId: string) => LayerCacheEntry | null;
  readonly setOverride: (layerId: string, transform: LayerTransform | null) => void;
  readonly replaceCache: (layerId: string, rect: Rect, surface: RasterSurface) => void;
  readonly restoreCache: (layerId: string, rect: Rect, pixels: ImageData) => void;
  readonly dispatch: (action: WorkbenchAction) => void;
  readonly pushHistory: (entry: HistoryEntry) => void;
  readonly canEdit: () => boolean;
  readonly isGestureActive: () => boolean;
  readonly endBurst: () => void;
  readonly invalidate: (payload: { layers: string[]; overlay: true }) => void;
}

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
    const layer = document?.layers.find((candidate) => candidate.id === layerId);
    if (!document || !layer || !layer.isEnabled || layer.isLocked || !hittableLayerSize(layer, document)) {
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

  private bakeEntry(
    layerId: string,
    beforeRect: Rect,
    before: ImageData,
    afterRect: Rect,
    after: ImageData,
    oldTransform: LayerTransform,
    newTransform: LayerTransform
  ): HistoryEntry {
    return {
      bytes: before.data.byteLength + after.data.byteLength + 256,
      label: 'Transform layer',
      redo: () => {
        this.deps.dispatch({ id: layerId, patch: { transform: newTransform }, type: 'updateCanvasLayer' });
        this.deps.restoreCache(layerId, afterRect, after);
      },
      undo: () => {
        this.deps.dispatch({ id: layerId, patch: { transform: oldTransform }, type: 'updateCanvasLayer' });
        this.deps.restoreCache(layerId, beforeRect, before);
      },
    };
  }

  apply(): void {
    if (this.disposed || !this.deps.canEdit() || this.deps.isGestureActive()) {
      return;
    }
    const session = this.deps.session.get();
    if (!session) {
      return;
    }
    const document = this.deps.getDocument();
    const layer = document?.layers.find((candidate) => candidate.id === session.layerId);
    const size = document && layer ? hittableLayerSize(layer, document) : null;
    if (
      !document ||
      !layer ||
      !size ||
      !isRenderableLayer(layer) ||
      layer.isLocked ||
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
      this.deps.endBurst();
      this.deps.setOverride(session.layerId, null);
      this.deps.session.set(null);
      const forward: WorkbenchAction = {
        id: session.layerId,
        patch: { transform: session.transform },
        type: 'updateCanvasLayer',
      };
      const inverse: WorkbenchAction = {
        id: session.layerId,
        patch: { transform: session.startTransform },
        type: 'updateCanvasLayer',
      };
      this.deps.dispatch(forward);
      this.deps.pushHistory(
        createDocumentPatchEntry({ dispatch: this.deps.dispatch, forward, inverse, label: 'Transform layer' })
      );
      this.deps.invalidate({ layers: [session.layerId], overlay: true });
      return;
    }
    if (source?.type !== 'paint') {
      this.cancel();
      return;
    }
    const cache = this.deps.getCache(layer.id);
    if (!cache || isEmpty(cache.rect)) {
      this.cancel();
      return;
    }
    this.deps.endBurst();
    const beforeRect = { ...cache.rect };
    const before = cache.surface.ctx.getImageData(0, 0, beforeRect.width, beforeRect.height);
    const matrix = bakeMatrix(session.transform);
    const afterRect = roundOut(transformBounds(matrix, beforeRect));
    const baked = this.deps.backend.createSurface(afterRect.width, afterRect.height);
    const context = baked.ctx;
    context.setTransform(1, 0, 0, 1, 0, 0);
    context.clearRect(0, 0, afterRect.width, afterRect.height);
    context.imageSmoothingEnabled = true;
    context.setTransform(matrix.a, matrix.b, matrix.c, matrix.d, matrix.e - afterRect.x, matrix.f - afterRect.y);
    context.drawImage(cache.surface.canvas, beforeRect.x, beforeRect.y);
    context.setTransform(1, 0, 0, 1, 0, 0);
    const after = context.getImageData(0, 0, afterRect.width, afterRect.height);
    const oldTransform = { ...session.startTransform };
    const identity: LayerTransform = { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 };
    this.deps.setOverride(layer.id, null);
    this.deps.session.set(null);
    this.deps.dispatch({ id: layer.id, patch: { transform: identity }, type: 'updateCanvasLayer' });
    this.deps.replaceCache(layer.id, afterRect, baked);
    this.deps.pushHistory(this.bakeEntry(layer.id, beforeRect, before, afterRect, after, oldTransform, identity));
  }

  dispose(): void {
    if (this.disposed) {
      return;
    }
    this.cancel();
    this.disposed = true;
  }
}
