import type { CanvasDocumentSnapshot, LayerExportGuard } from '@workbench/canvas-engine/capabilities';
import type {
  CanvasDocumentContractV2,
  CanvasLayerContract,
  CanvasStateContractV2,
} from '@workbench/canvas-engine/contracts';
import type { ExportLayerPixelsResult } from '@workbench/canvas-engine/controllers/rasterExportController';
import type { RasterMemoryBudgetController } from '@workbench/canvas-engine/controllers/rasterMemoryBudgetController';
import type { CanvasRasterSnapshot, CaptureRasterSnapshotResult } from '@workbench/canvas-engine/rasterTransactions';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Rect } from '@workbench/canvas-engine/types';

import { getSourceContentRect, renderableSourceOf } from '@workbench/canvas-engine/document/sources';
import { isSupportedExportSource } from '@workbench/canvas-engine/layerExportGuards';

const BYTES_PER_PIXEL = 4;

/**
 * What a layer is expected to cost once rasterized, from its source rather than
 * from a raster pass. Reserved up front so a capture that cannot fit is refused
 * before any work is done; the real cost is topped up per layer afterwards.
 */
const estimatedLayerBytes = (layer: CanvasLayerContract, document: CanvasDocumentContractV2): number => {
  const source = renderableSourceOf(layer);
  if (source?.type === 'image') {
    return source.image.width * source.image.height * BYTES_PER_PIXEL;
  }
  if (source?.type === 'paint' && source.bitmap) {
    return source.bitmap.width * source.bitmap.height * BYTES_PER_PIXEL;
  }
  const contentRect = getSourceContentRect(layer, document);
  return contentRect.width * contentRect.height * BYTES_PER_PIXEL;
};

export interface CreateRasterSnapshotCaptureDeps {
  readonly memory: RasterMemoryBudgetController;
  /** Re-reads the live cache sizes into the budget before a reservation decision. */
  readonly syncMemoryBaselines: () => void;
  readonly createSurface: (width: number, height: number) => RasterSurface;
  readonly getCanvasState: () => CanvasStateContractV2 | null;
  readonly getDocumentGeneration: () => number;
  readonly getContentEpoch: () => number;
  readonly getLifecycleGeneration: () => number;
  readonly isDisposed: () => boolean;
  readonly isGuardCurrent: (guard: LayerExportGuard) => boolean;
  readonly rasterizeLayerPixels: (
    layerId: string,
    options: { includeDisabled?: boolean; signal?: AbortSignal }
  ) => Promise<ExportLayerPixelsResult>;
}

export interface RasterSnapshotCapture {
  /** Clones the current canvas state and stamps it with the engine state it came from. */
  captureDocumentSnapshot(): CanvasDocumentSnapshot | null;
  /** Whether the engine has moved on from the state a snapshot was taken against. */
  isDocumentSnapshotCurrent(snapshot: CanvasDocumentSnapshot): boolean;
  /** Rasterizes the named layers into detached surfaces owned by the returned snapshot. */
  captureRasterSnapshot(
    documentSnapshot: CanvasDocumentSnapshot,
    layerIds: readonly string[],
    options?: { signal?: AbortSignal; includeDisabled?: boolean }
  ): Promise<CaptureRasterSnapshotResult>;
  /** Releases every snapshot still held, for engine teardown. */
  releaseActiveSnapshots(): void;
}

/**
 * Takes point-in-time copies of the canvas for long-running background work —
 * PSD export, invocation composites — that must not see the document change
 * underneath it.
 *
 * A document snapshot is a structural clone plus a private record of the engine
 * state it was cloned from: the canvas object identity, the raster content
 * epoch, the lifecycle generation, and the document generation. Currency is
 * that whole tuple, so an edit that produces an identical-looking document
 * still invalidates the snapshot. The record lives in a `WeakMap` keyed by the
 * snapshot, which is also how a foreign object is rejected as `'not-ready'`
 * rather than mistaken for a stale one.
 *
 * A raster snapshot then detaches real pixels. Because rasterizing is async and
 * one capture covers many layers, currency is re-checked before and after every
 * layer, and every guard captured along the way is re-checked once more before
 * the snapshot is published — a capture that raced an edit is discarded whole
 * rather than returning a mix of old and new pixels. Bytes are reserved up
 * front from a source-based estimate and topped up per layer when a rasterized
 * surface turns out larger, so a capture cannot overrun the budget mid-flight.
 */
export const createRasterSnapshotCapture = (deps: CreateRasterSnapshotCaptureDeps): RasterSnapshotCapture => {
  const { isGuardCurrent, memory, syncMemoryBaselines } = deps;

  const activeSnapshots = new Set<CanvasRasterSnapshot>();
  const snapshotSources = new WeakMap<
    CanvasDocumentSnapshot,
    { canvas: CanvasStateContractV2; contentEpoch: number; lifecycleGeneration: number }
  >();

  const isDocumentSnapshotCurrent = (snapshot: CanvasDocumentSnapshot): boolean => {
    const source = snapshotSources.get(snapshot);
    return (
      !deps.isDisposed() &&
      source !== undefined &&
      source.canvas === deps.getCanvasState() &&
      source.contentEpoch === deps.getContentEpoch() &&
      source.lifecycleGeneration === deps.getLifecycleGeneration() &&
      snapshot.documentGeneration === deps.getDocumentGeneration()
    );
  };

  const reserveBytes = (bytes: number, generation: number): ReturnType<RasterMemoryBudgetController['reserve']> => {
    syncMemoryBaselines();
    return memory.reserve(bytes, { generation, purpose: 'background-snapshot' });
  };

  const captureRasterSnapshot = async (
    documentSnapshot: CanvasDocumentSnapshot,
    layerIds: readonly string[],
    options?: { signal?: AbortSignal; includeDisabled?: boolean }
  ): Promise<CaptureRasterSnapshotResult> => {
    if (options?.signal?.aborted) {
      return { status: 'aborted' };
    }
    if (!snapshotSources.has(documentSnapshot)) {
      return { status: 'not-ready' };
    }
    if (!isDocumentSnapshotCurrent(documentSnapshot)) {
      return { status: 'stale' };
    }
    const captureLifecycleGeneration = snapshotSources.get(documentSnapshot)!.lifecycleGeneration;

    const document = documentSnapshot.canvas.document;
    const uniqueLayerIds = [...new Set(layerIds)];
    const layerById = new Map(document.layers.map((layer) => [layer.id, layer]));
    let requestedBytes = 0;
    for (const layerId of uniqueLayerIds) {
      const layer = layerById.get(layerId);
      const source = layer ? renderableSourceOf(layer) : null;
      if (!layer || !source || !isSupportedExportSource(source)) {
        return { status: 'not-ready' };
      }
      requestedBytes += estimatedLayerBytes(layer, document);
    }

    const reservation = reserveBytes(requestedBytes, captureLifecycleGeneration);
    if (reservation.status === 'over-budget') {
      return { status: 'over-budget' };
    }
    const reservationLeases: { release(): void }[] = [reservation.lease];
    const pinLeases = uniqueLayerIds.map((layerId) => memory.pin(layerId, captureLifecycleGeneration));
    const layerSurfaces = new Map<string, { rect: Rect; surface: RasterSurface }>();
    const emptyLayerIds = new Set<string>();
    const capturedGuards: LayerExportGuard[] = [];
    let actualDetachedBytes = 0;
    try {
      for (const layerId of uniqueLayerIds) {
        if (options?.signal?.aborted) {
          return { status: 'aborted' };
        }
        if (!isDocumentSnapshotCurrent(documentSnapshot)) {
          return { status: 'stale' };
        }
        const liveResult = await deps.rasterizeLayerPixels(layerId, {
          includeDisabled: options?.includeDisabled,
          signal: options?.signal,
        });
        if (options?.signal?.aborted) {
          return { status: 'aborted' };
        }
        if (!isDocumentSnapshotCurrent(documentSnapshot)) {
          return { status: 'stale' };
        }
        if (liveResult.status === 'empty') {
          emptyLayerIds.add(layerId);
          continue;
        }
        if (liveResult.status !== 'ok') {
          return {
            status:
              liveResult.status === 'aborted' || liveResult.status === 'over-budget' ? liveResult.status : 'not-ready',
          };
        }
        const live = liveResult;
        if (!isGuardCurrent(live.guard)) {
          live.release();
          return { status: 'stale' };
        }
        capturedGuards.push(live.guard);
        try {
          const actualBytes = live.surface.width * live.surface.height * BYTES_PER_PIXEL;
          // The up-front reservation was an estimate from the source; a surface
          // that rasterized larger has to be paid for before it is detached.
          const additionalBytes = Math.max(0, actualBytes - estimatedLayerBytes(layerById.get(layerId)!, document));
          if (additionalBytes > 0) {
            const additional = reserveBytes(additionalBytes, captureLifecycleGeneration);
            if (additional.status === 'over-budget') {
              return { status: 'over-budget' };
            }
            reservationLeases.push(additional.lease);
          }
          const detached = deps.createSurface(live.surface.width, live.surface.height);
          detached.ctx.setTransform(1, 0, 0, 1, 0, 0);
          detached.ctx.clearRect(0, 0, detached.width, detached.height);
          detached.ctx.drawImage(live.surface.canvas, 0, 0);
          layerSurfaces.set(layerId, { rect: { ...live.rect }, surface: detached });
          actualDetachedBytes += actualBytes;
        } finally {
          live.release();
        }
      }
      // One last check across the whole capture: an edit part-way through leaves
      // earlier layers describing a document the later ones no longer match.
      if (!isDocumentSnapshotCurrent(documentSnapshot) || capturedGuards.some((guard) => !isGuardCurrent(guard))) {
        return { status: 'stale' };
      }

      // The reservations covered the capture; the published snapshot is tracked
      // as detached bytes instead, for as long as the caller holds it.
      for (const lease of reservationLeases) {
        lease.release();
      }
      const detachedLease = memory.trackDetached(actualDetachedBytes, captureLifecycleGeneration);
      let released = false;
      const snapshot: CanvasRasterSnapshot = {
        ...documentSnapshot,
        emptyLayerIds,
        layerSurfaces,
        release: () => {
          if (released) {
            return;
          }
          released = true;
          emptyLayerIds.clear();
          layerSurfaces.clear();
          detachedLease.release();
          activeSnapshots.delete(snapshot);
        },
      };
      activeSnapshots.add(snapshot);
      return { snapshot, status: 'ok' };
    } finally {
      for (const lease of reservationLeases) {
        lease.release();
      }
      for (const lease of pinLeases) {
        lease.release();
      }
    }
  };

  return {
    captureDocumentSnapshot: () => {
      const canvas = deps.getCanvasState();
      if (deps.isDisposed() || !canvas) {
        return null;
      }
      const snapshot: CanvasDocumentSnapshot = {
        canvas: structuredClone(canvas),
        documentGeneration: deps.getDocumentGeneration(),
      };
      snapshotSources.set(snapshot, {
        canvas,
        contentEpoch: deps.getContentEpoch(),
        lifecycleGeneration: deps.getLifecycleGeneration(),
      });
      return snapshot;
    },

    captureRasterSnapshot,

    isDocumentSnapshotCurrent,

    releaseActiveSnapshots: () => {
      for (const snapshot of activeSnapshots) {
        snapshot.release();
      }
    },
  };
};
