import type { DuplicateLayersResult } from '@workbench/canvas-engine/capabilities';
import type { CanvasDocumentContractV2, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { RasterMemoryReservationResult } from '@workbench/canvas-engine/controllers/rasterMemoryBudgetController';
import type { History } from '@workbench/canvas-engine/history/history';
import type { CanvasProjectMutation } from '@workbench/canvas-engine/mutationContracts';
import type { PreparedLayerCacheReplacement } from '@workbench/canvas-engine/render/layerCache';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Rect } from '@workbench/canvas-engine/types';

export type CapturedLayerCache = { pixels: RasterSurface; rect: Rect } | null | 'not-ready';

export interface LayerMutationControllerOptions {
  readonly canEdit: () => boolean;
  readonly captureCache: (layer: CanvasLayerContract, document: CanvasDocumentContractV2) => CapturedLayerCache;
  readonly createLayerId: () => string;
  readonly discardPersisted: (layerId: string) => void;
  readonly dispatchPrepared: (
    action: CanvasProjectMutation,
    reducerAccepted: () => boolean,
    mirrorAccepted: () => boolean
  ) => void;
  readonly endBurst: () => void;
  readonly estimateCacheBytes: (layer: CanvasLayerContract, document: CanvasDocumentContractV2) => number | 'not-ready';
  readonly getDocument: () => CanvasDocumentContractV2 | null;
  readonly getReducerDocument: () => CanvasDocumentContractV2 | null;
  readonly getSelectedLayerIds: (document: CanvasDocumentContractV2) => readonly string[];
  readonly history: History;
  readonly installPrepared: (prepared: PreparedLayerCacheReplacement, persist?: boolean) => void;
  readonly isGestureActive: () => boolean;
  readonly needsPixelPersistence: (layer: CanvasLayerContract) => boolean;
  readonly preparePixels: (layerId: string, rect: Rect, pixels: RasterSurface) => PreparedLayerCacheReplacement;
  readonly publishSelectedLayerIds: (primaryId: string | null, selectedIds: readonly string[]) => void;
  readonly reserve: (bytes: number) => RasterMemoryReservationResult;
  readonly sameContract: (document: CanvasDocumentContractV2 | null, layer: CanvasLayerContract) => boolean;
}

/** Owns failure-atomic copy and cross-type conversion mutations. */
export class LayerMutationController {
  constructor(private readonly options: LayerMutationControllerOptions) {}

  duplicate(layerIds: readonly string[]): DuplicateLayersResult | null {
    const o = this.options;
    if (!o.canEdit() || o.isGestureActive()) {
      return null;
    }
    o.endBurst();
    const document = o.getDocument();
    if (!document) {
      return null;
    }
    const requested = new Set(layerIds);
    const sources = document.layers.filter((layer) => requested.has(layer.id));
    if (sources.length === 0 || sources.length !== requested.size) {
      return null;
    }
    const estimates = sources.map((source) => o.estimateCacheBytes(source, document));
    let capturedBytes = 0;
    for (const estimate of estimates) {
      if (estimate === 'not-ready') {
        return null;
      }
      capturedBytes += estimate;
    }
    const historyBytes = capturedBytes + sources.length * 256;
    if (!o.history.canRetain(historyBytes)) {
      return null;
    }
    // The initial transaction owns one immutable history capture and one
    // prepared replacement per populated source at its allocation peak.
    const reservation = o.reserve(capturedBytes * 2);
    if (reservation.status === 'over-budget') {
      return null;
    }
    try {
      const captures = sources.map((source) => o.captureCache(source, document));
      if (captures.some((capture) => capture === 'not-ready')) {
        return null;
      }
      const existingIds = new Set(document.layers.map((layer) => layer.id));
      const duplicates = sources.map((source) => {
        const duplicate = structuredClone(source);
        duplicate.id = o.createLayerId();
        duplicate.name = `${source.name} copy`;
        return duplicate;
      });
      if (
        duplicates.some((layer) => existingIds.has(layer.id)) ||
        new Set(duplicates.map((layer) => layer.id)).size !== duplicates.length
      ) {
        return null;
      }
      const duplicateBySource = new Map(sources.map((source, index) => [source.id, duplicates[index]!]));
      const orderedIds = document.layers.flatMap((layer) => {
        const duplicate = duplicateBySource.get(layer.id);
        return duplicate ? [duplicate.id, layer.id] : [layer.id];
      });
      const selectedLayerId =
        (document.selectedLayerId ? duplicateBySource.get(document.selectedLayerId)?.id : null) ?? duplicates[0]!.id;
      const previousSelectedLayerId = document.selectedLayerId;
      const previousSelectedIds = [...o.getSelectedLayerIds(document)];
      const originalIds = document.layers.map((layer) => layer.id);
      const duplicateIds = duplicates.map((layer) => layer.id);
      const hasDuplicates = (candidate: CanvasDocumentContractV2 | null): boolean =>
        candidate?.selectedLayerId === selectedLayerId &&
        candidate.layers.length === orderedIds.length &&
        candidate.layers.every((layer, index) => layer.id === orderedIds[index]);
      const hasOriginals = (candidate: CanvasDocumentContractV2 | null): boolean =>
        candidate?.selectedLayerId === previousSelectedLayerId &&
        candidate.layers.length === originalIds.length &&
        candidate.layers.every((layer, index) => layer.id === originalIds[index]);
      const applyPrepared = (): void => {
        const prepared = captures.flatMap((capture, index) => {
          if (!capture || capture === 'not-ready') {
            return [];
          }
          const duplicate = duplicates[index]!;
          return [{ duplicate, replacement: o.preparePixels(duplicate.id, capture.rect, capture.pixels) }];
        });
        o.dispatchPrepared(
          {
            add: { index: 0, layers: duplicates },
            enabledUpdates: [],
            orderedIds,
            selectedLayerId,
            type: 'applyCanvasLayerStackMutation',
          },
          () => hasDuplicates(o.getReducerDocument()),
          () => hasDuplicates(o.getDocument())
        );
        prepared.forEach(({ duplicate, replacement }) => {
          o.installPrepared(replacement, o.needsPixelPersistence(duplicate));
        });
        o.publishSelectedLayerIds(selectedLayerId, duplicateIds);
      };
      const redo = (): void => {
        const replayReservation = o.reserve(capturedBytes);
        if (replayReservation.status === 'over-budget') {
          throw new Error('Not enough raster memory to restore duplicated layers');
        }
        try {
          applyPrepared();
        } finally {
          replayReservation.lease.release();
        }
      };
      applyPrepared();
      o.history.push({
        bytes: historyBytes,
        label: duplicates.length === 1 ? 'Duplicate layer' : 'Duplicate layers',
        redo,
        replayFailureAtomic: true,
        undo: () => {
          o.dispatchPrepared(
            {
              enabledUpdates: [],
              removeIds: duplicateIds,
              selectedLayerId: previousSelectedLayerId,
              type: 'applyCanvasLayerStackMutation',
            },
            () => hasOriginals(o.getReducerDocument()),
            () => hasOriginals(o.getDocument())
          );
          o.publishSelectedLayerIds(previousSelectedLayerId, previousSelectedIds);
        },
      });
      return { duplicateIds, selectedLayerId };
    } finally {
      reservation.lease.release();
    }
  }

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
