import type { CanvasDocumentContractV2, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { History } from '@workbench/canvas-engine/history/history';
import type { PreparedLayerCacheReplacement } from '@workbench/canvas-engine/render/layerCache';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Rect } from '@workbench/canvas-engine/types';
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';

export type CapturedLayerCache = { pixels: RasterSurface; rect: Rect } | null | 'not-ready';

export interface LayerMutationControllerOptions {
  readonly canEdit: () => boolean;
  readonly captureCache: (layer: CanvasLayerContract, document: CanvasDocumentContractV2) => CapturedLayerCache;
  readonly discardPersisted: (layerId: string) => void;
  readonly dispatchPrepared: (
    action: CanvasProjectMutation,
    reducerAccepted: () => boolean,
    mirrorAccepted: () => boolean
  ) => void;
  readonly endBurst: () => void;
  readonly getDocument: () => CanvasDocumentContractV2 | null;
  readonly getReducerDocument: () => CanvasDocumentContractV2 | null;
  readonly history: History;
  readonly installPrepared: (prepared: PreparedLayerCacheReplacement, persist?: boolean) => void;
  readonly isGestureActive: () => boolean;
  readonly needsPixelPersistence: (layer: CanvasLayerContract) => boolean;
  readonly preparePixels: (layerId: string, rect: Rect, pixels: RasterSurface) => PreparedLayerCacheReplacement;
  readonly sameContract: (document: CanvasDocumentContractV2 | null, layer: CanvasLayerContract) => boolean;
}

/** Owns failure-atomic copy and cross-type conversion mutations. */
export class LayerMutationController {
  constructor(private readonly options: LayerMutationControllerOptions) {}

  copy(label: string, sourceLayerId: string, layer: CanvasLayerContract, index: number): boolean {
    const o = this.options;
    if (!o.canEdit() || o.isGestureActive()) {
      return false;
    }
    o.endBurst();
    const document = o.getDocument();
    const source = document?.layers.find((candidate) => candidate.id === sourceLayerId);
    if (!document || !source || document.layers.some((candidate) => candidate.id === layer.id)) {
      return false;
    }
    const captured = o.captureCache(source, document);
    if (captured === 'not-ready') {
      return false;
    }
    const selectedLayerId = document.selectedLayerId;
    const apply = (): void => {
      const prepared = captured ? o.preparePixels(layer.id, captured.rect, captured.pixels) : null;
      o.dispatchPrepared(
        {
          add: { index, layers: [layer] },
          enabledUpdates: [],
          selectedLayerId: layer.id,
          type: 'applyCanvasLayerStackMutation',
        },
        () =>
          o.getReducerDocument()?.selectedLayerId === layer.id &&
          o.getReducerDocument()?.layers.some((candidate) => candidate === layer) === true,
        () =>
          o.getDocument()?.selectedLayerId === layer.id &&
          o.getDocument()?.layers.some((candidate) => candidate === layer) === true
      );
      if (prepared) {
        o.installPrepared(prepared, o.needsPixelPersistence(layer));
      }
    };
    apply();
    o.history.push({
      bytes: captured ? captured.rect.width * captured.rect.height * 4 + 256 : 256,
      label,
      redo: apply,
      replayFailureAtomic: true,
      undo: () =>
        o.dispatchPrepared(
          { enabledUpdates: [], removeIds: [layer.id], selectedLayerId, type: 'applyCanvasLayerStackMutation' },
          () =>
            o.getReducerDocument()?.selectedLayerId === selectedLayerId &&
            o.getReducerDocument()?.layers.some((candidate) => candidate.id === layer.id) === false,
          () =>
            o.getDocument()?.selectedLayerId === selectedLayerId &&
            o.getDocument()?.layers.some((candidate) => candidate.id === layer.id) === false
        ),
    });
    return true;
  }

  convert(label: string, expected: CanvasLayerContract, after: CanvasLayerContract): boolean {
    const o = this.options;
    if (!o.canEdit() || o.isGestureActive() || expected.id !== after.id || expected.type === after.type) {
      return false;
    }
    o.endBurst();
    const document = o.getDocument();
    const current = document?.layers.find((candidate) => candidate.id === expected.id);
    if (!document || !current || current !== expected || current.isLocked || current.type !== expected.type) {
      return false;
    }
    const captured = o.captureCache(current, document);
    if (captured === 'not-ready') {
      return false;
    }
    const apply = (layer: CanvasLayerContract): void => {
      const prepared = captured ? o.preparePixels(layer.id, captured.rect, captured.pixels) : null;
      o.dispatchPrepared(
        { id: layer.id, layer, targetType: layer.type, type: 'convertCanvasLayer' },
        () => o.sameContract(o.getReducerDocument(), layer),
        () => o.sameContract(o.getDocument(), layer)
      );
      try {
        o.discardPersisted(layer.id);
      } catch {
        /* Ancillary after reducer acceptance. */
      }
      if (prepared) {
        o.installPrepared(prepared, o.needsPixelPersistence(layer));
      }
    };
    const before = structuredClone(current);
    apply(after);
    o.history.push({
      bytes: captured ? captured.rect.width * captured.rect.height * 4 + 256 : 256,
      label,
      redo: () => apply(after),
      replayFailureAtomic: true,
      undo: () => apply(before),
    });
    return true;
  }

  dispose(): void {}
}
