import type { CanvasDocumentContractV3 } from '@workbench/canvas-engine/contracts';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Rect } from '@workbench/canvas-engine/types';

import { isEmpty } from '@workbench/canvas-engine/math/rect';
import {
  type AdjustmentRefusal,
  AdjustmentRefusedError,
  adjustmentRefusal,
} from '@workbench/canvas-engine/render/exactAdjustments';
import {
  compositeScratchPixels,
  getBaseRasterContentBounds,
  planBaseRasterComposite,
  RasterCompositeOverBudgetError,
  renderRasterComposite,
  type RenderRasterCompositeDeps,
} from '@workbench/canvas-engine/render/rasterComposite';

export type RasterCompositeExportRequest = { bounds: 'content' } | { bounds: 'rect'; rect: Rect };

export type RasterCompositeExportResult =
  | { status: 'ok'; blob: Blob; rect: Rect }
  | { status: 'empty' | 'stale' | 'not-ready' | 'over-budget' }
  | AdjustmentRefusal;

export interface RasterCompositeExportSnapshot {
  directPixelEpoch: number;
  document: CanvasDocumentContractV3 | null;
  documentGeneration: number;
  lifecycleGeneration: number;
}

export interface ExportRasterCompositeDeps extends RenderRasterCompositeDeps {
  backend: RenderRasterCompositeDeps['backend'] & {
    encodeSurface(surface: RasterSurface, type?: string): Promise<Blob>;
  };
  captureSnapshot(): RasterCompositeExportSnapshot;
  isSnapshotCurrent(snapshot: RasterCompositeExportSnapshot): boolean;
  pin?(layerIds: readonly string[]): { release(): void };
}

export const exportRasterComposite = async (
  request: RasterCompositeExportRequest,
  deps: ExportRasterCompositeDeps
): Promise<RasterCompositeExportResult> => {
  const snapshot = deps.captureSnapshot();
  const document = snapshot.document;
  if (!document) {
    return { status: 'not-ready' };
  }

  const rect = request.bounds === 'content' ? getBaseRasterContentBounds(document) : request.rect;
  if (!rect || isEmpty(rect)) {
    return { status: 'empty' };
  }

  const entry = planBaseRasterComposite(document, rect);
  if (entry.layers.length === 0) {
    return { status: 'empty' };
  }

  // The final composite surface, plus each adjusted layer's and group scope's scratch surface and its ImageData.
  const reservation = deps.reserve?.((rect.width * rect.height + compositeScratchPixels(entry) * 2) * 4);
  if (reservation?.status === 'over-budget') {
    return { status: 'over-budget' };
  }
  const pinLease = deps.pin?.(entry.layers.map((layer) => layer.id));

  try {
    let surface: RasterSurface;
    try {
      surface = await renderRasterComposite(entry, deps);
    } catch (error) {
      if (error instanceof RasterCompositeOverBudgetError) {
        return { status: 'over-budget' };
      }
      if (error instanceof AdjustmentRefusedError) {
        return adjustmentRefusal(document, error.ownerId, error.reason);
      }
      // Spatial work is cancelled only by cooldown, cache release or disposal; the export can simply be retried.
      if (!deps.isSnapshotCurrent(snapshot) || (error instanceof DOMException && error.name === 'AbortError')) {
        return { status: 'stale' };
      }
      throw error;
    }
    if (!deps.isSnapshotCurrent(snapshot)) {
      return { status: 'stale' };
    }

    const blob = await deps.backend.encodeSurface(surface, 'image/png');
    return deps.isSnapshotCurrent(snapshot) ? { status: 'ok', blob, rect } : { status: 'stale' };
  } finally {
    pinLease?.release();
    if (reservation?.status === 'ok') {
      reservation.lease.release();
    }
  }
};
