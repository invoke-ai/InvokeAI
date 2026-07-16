import type {
  CanvasDetachedLayerSurface,
  CanvasDocumentSnapshot,
  CaptureRasterSnapshotResult,
  PsdExportResult,
} from '@workbench/canvas-engine/api';
import type { RasterBackend } from '@workbench/canvas-engine/render/raster';
import type { CanvasLayerContract } from '@workbench/types';

import {
  executePsdExport,
  planPsdExport,
  type ExecutePsdExportDeps,
  type PsdExportLayerInput,
  type PsdExportPlan,
} from '@workbench/canvas-engine/export/psdExport';

type RasterMemoryReservationResult =
  | { status: 'ok'; lease: { release(): void } }
  | { status: 'over-budget'; requestedBytes: number; availableBytes: number };

export interface PsdExportControllerOptions {
  readonly backend: RasterBackend;
  readonly captureDocumentSnapshot: () => CanvasDocumentSnapshot | null;
  readonly captureRasterSnapshot: (
    snapshot: CanvasDocumentSnapshot,
    layerIds: readonly string[],
    options: { signal: AbortSignal; includeDisabled: boolean }
  ) => Promise<CaptureRasterSnapshotResult>;
  readonly execute?: (plan: PsdExportPlan, fileName: string, deps: ExecutePsdExportDeps) => Promise<void>;
  readonly getAvailableBytes: () => number;
  readonly isDocumentSnapshotCurrent: (snapshot: CanvasDocumentSnapshot) => boolean;
  readonly reserve: (bytes: number) => RasterMemoryReservationResult;
}

const isExportable = (layer: CanvasLayerContract): boolean => {
  if (layer.type !== 'raster') {
    return false;
  }
  switch (layer.source.type) {
    case 'paint':
    case 'image':
    case 'gradient':
    case 'text':
      return true;
    case 'shape':
      return layer.source.kind !== 'polygon';
    default:
      return false;
  }
};

export const PSD_ALLOCATION_BYTES_PER_PIXEL = 8;

export const derivePsdPixelAreaLimit = (availableBytes: number): number =>
  Math.max(0, Math.floor(availableBytes / PSD_ALLOCATION_BYTES_PER_PIXEL));

const getRequiredAllocationPixelArea = (plan: Extract<PsdExportPlan, { status: 'ok' }>): number =>
  plan.width * plan.height +
  plan.layers.reduce((total, layer) => total + layer.worldRect.width * layer.worldRect.height, 0);

/** Owns immutable PSD snapshot capture, budget reservation, execution, and cancellation. */
export class PsdExportController {
  private disposed = false;
  private readonly active = new Set<AbortController>();

  constructor(private readonly deps: PsdExportControllerOptions) {}

  async export(fileName: string, options: { signal?: AbortSignal } = {}): Promise<PsdExportResult> {
    if (this.disposed || options.signal?.aborted) {
      return 'aborted';
    }
    const abortController = new AbortController();
    const abort = (): void => abortController.abort(options.signal?.reason);
    options.signal?.addEventListener('abort', abort, { once: true });
    this.active.add(abortController);
    try {
      const documentSnapshot = this.deps.captureDocumentSnapshot();
      if (!documentSnapshot) {
        return 'nothing';
      }
      if (!this.deps.isDocumentSnapshotCurrent(documentSnapshot)) {
        return 'stale';
      }
      const document = documentSnapshot.canvas.document;
      const layers = document.layers.filter(isExportable);
      if (layers.length === 0) {
        return 'nothing';
      }
      const capture = await this.deps.captureRasterSnapshot(
        documentSnapshot,
        layers.map((layer) => layer.id),
        { includeDisabled: true, signal: abortController.signal }
      );
      if (capture.status !== 'ok') {
        return capture.status;
      }
      const rasterSnapshot = capture.snapshot;
      try {
        if (abortController.signal.aborted) {
          return 'aborted';
        }
        if (!this.deps.isDocumentSnapshotCurrent(documentSnapshot)) {
          return 'stale';
        }
        const inputs: PsdExportLayerInput[] = [];
        for (const layer of layers) {
          const detached = rasterSnapshot.layerSurfaces.get(layer.id);
          if (!detached) {
            if (rasterSnapshot.emptyLayerIds.has(layer.id)) {
              continue;
            }
            return 'not-ready';
          }
          inputs.push({
            adjustments: layer.type === 'raster' ? layer.adjustments : undefined,
            blendMode: layer.blendMode,
            contentRect: detached.rect,
            id: layer.id,
            isEnabled: layer.isEnabled,
            name: layer.name,
            opacity: layer.opacity,
            transform: layer.transform,
          });
        }
        const plan = planPsdExport(inputs);
        if (plan.status === 'empty') {
          return 'nothing';
        }
        if (plan.status === 'too-large') {
          return 'too-large';
        }
        const requestedPixelArea = getRequiredAllocationPixelArea(plan);
        const requestedBytes = requestedPixelArea * PSD_ALLOCATION_BYTES_PER_PIXEL;
        if (requestedPixelArea > derivePsdPixelAreaLimit(this.deps.getAvailableBytes())) {
          return 'over-budget';
        }
        const reservation = this.deps.reserve(requestedBytes);
        if (reservation.status === 'over-budget') {
          return 'over-budget';
        }
        try {
          const getLayerSurface = (layerId: string): Promise<CanvasDetachedLayerSurface> => {
            const detached = rasterSnapshot.layerSurfaces.get(layerId);
            if (!detached) {
              return Promise.reject(new Error(`PSD raster snapshot is missing layer ${layerId}.`));
            }
            return Promise.resolve(detached);
          };
          try {
            await (this.deps.execute ?? executePsdExport)(
              plan,
              /\.psd$/i.test(fileName) ? fileName : `${fileName}.psd`,
              { backend: this.deps.backend, getLayerSurface, signal: abortController.signal }
            );
          } catch (error) {
            if (abortController.signal.aborted || (error instanceof DOMException && error.name === 'AbortError')) {
              return 'aborted';
            }
            throw error;
          }
          return abortController.signal.aborted ? 'aborted' : 'exported';
        } finally {
          reservation.lease.release();
        }
      } finally {
        rasterSnapshot.release();
      }
    } finally {
      this.active.delete(abortController);
      options.signal?.removeEventListener('abort', abort);
    }
  }

  cancel(): void {
    for (const controller of this.active) {
      controller.abort();
    }
  }

  dispose(): void {
    if (this.disposed) {
      return;
    }
    this.disposed = true;
    this.cancel();
  }
}
