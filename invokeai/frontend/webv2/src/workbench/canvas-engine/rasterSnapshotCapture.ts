import type { CanvasDocumentSnapshot, LayerExportGuard } from '@workbench/canvas-engine/capabilities';
import type {
  CanvasDocumentContractV3,
  CanvasLayerContract,
  CanvasStateContractV3,
} from '@workbench/canvas-engine/contracts';
import type { ExportLayerPixelsResult } from '@workbench/canvas-engine/controllers/rasterExportController';
import type { RasterMemoryBudgetController } from '@workbench/canvas-engine/controllers/rasterMemoryBudgetController';
import type { CanvasRasterSnapshot, CaptureRasterSnapshotResult } from '@workbench/canvas-engine/rasterTransactions';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Rect } from '@workbench/canvas-engine/types';

import { lookupDocumentLayer } from '@workbench/canvas-engine/document-model/documentModel';
import { getSourceContentRect, renderableSourceOf } from '@workbench/canvas-engine/document/sources';
import { isSupportedExportSource } from '@workbench/canvas-engine/layerExportGuards';

const BYTES_PER_PIXEL = 4;

/** Reserve source-estimated raster bytes before capture; top up with actual layer costs after rasterization. */
const estimatedLayerBytes = (layer: CanvasLayerContract, document: CanvasDocumentContractV3): number => {
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
  readonly memory: Pick<RasterMemoryBudgetController, 'pin' | 'reserve' | 'trackDetached'>;
  readonly createSurface: (width: number, height: number) => RasterSurface;
  readonly getCanvasState: () => CanvasStateContractV3 | null;
  readonly getDocumentGeneration: () => number;
  readonly getDirectPixelEpoch: () => number;
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
 * Document snapshots clone structure and privately record canvas identity, raster epoch and lifecycle/document
 * generations. WeakMap membership distinguishes foreign from stale snapshots.
 *
 * Raster capture checks currency before/after every layer and all guards before publication, rejecting mixed
 * revisions. Reserve estimated bytes upfront and top up larger actual surfaces before exceeding budget.
 */
export const createRasterSnapshotCapture = (deps: CreateRasterSnapshotCaptureDeps): RasterSnapshotCapture => {
  const { isGuardCurrent, memory } = deps;

  const activeSnapshots = new Set<CanvasRasterSnapshot>();
  const snapshotSources = new WeakMap<
    CanvasDocumentSnapshot,
    { canvas: CanvasStateContractV3; directPixelEpoch: number; lifecycleGeneration: number }
  >();

  const isDocumentSnapshotCurrent = (snapshot: CanvasDocumentSnapshot): boolean => {
    const source = snapshotSources.get(snapshot);
    return (
      !deps.isDisposed() &&
      source !== undefined &&
      source.canvas === deps.getCanvasState() &&
      source.directPixelEpoch === deps.getDirectPixelEpoch() &&
      source.lifecycleGeneration === deps.getLifecycleGeneration() &&
      snapshot.documentGeneration === deps.getDocumentGeneration()
    );
  };

  const reserveBytes = (bytes: number, generation: number): ReturnType<RasterMemoryBudgetController['reserve']> =>
    memory.reserve(bytes, { generation, purpose: 'background-snapshot' });

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
    let requestedBytes = 0;
    for (const layerId of uniqueLayerIds) {
      const layer = lookupDocumentLayer(document, layerId);
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
    // Earlier layers stay resident while later ones rasterize, so the final guard check cannot fail spuriously.
    const pinLeases = uniqueLayerIds.map((layerId) => memory.pin(layerId));
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
          const snapshotLayer = lookupDocumentLayer(document, layerId)!;
          const additionalBytes = Math.max(0, actualBytes - estimatedLayerBytes(snapshotLayer, document));
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
      const detachedLease = memory.trackDetached(actualDetachedBytes);
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
        directPixelEpoch: deps.getDirectPixelEpoch(),
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
