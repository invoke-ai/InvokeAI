import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Rect } from '@workbench/canvas-engine/types';
import type { CanvasDocumentContractV2 } from '@workbench/types';

import { isEmpty } from '@workbench/canvas-engine/math/rect';
import {
  getBaseRasterContentBounds,
  planBaseRasterComposite,
  renderRasterComposite,
  type RenderRasterCompositeDeps,
} from '@workbench/canvas-engine/render/rasterComposite';

export type RasterCompositeExportRequest = { bounds: 'content' } | { bounds: 'rect'; rect: Rect };

export type RasterCompositeExportResult =
  | { status: 'ok'; blob: Blob; rect: Rect }
  | { status: 'empty' | 'stale' | 'not-ready' | 'over-budget' };

export interface RasterCompositeExportSnapshot {
  contentEpoch: number;
  document: CanvasDocumentContractV2 | null;
  documentGeneration: number;
  lifecycleGeneration: number;
}

export class RasterCompositeOverBudgetError extends Error {
  constructor() {
    super('Raster composite allocation is over budget.');
    this.name = 'RasterCompositeOverBudgetError';
  }
}

export interface ExportRasterCompositeDeps extends RenderRasterCompositeDeps {
  backend: RenderRasterCompositeDeps['backend'] & {
    encodeSurface(surface: RasterSurface, type?: string): Promise<Blob>;
  };
  captureSnapshot(): RasterCompositeExportSnapshot;
  isSnapshotCurrent(snapshot: RasterCompositeExportSnapshot): boolean;
  reserve?(
    bytes: number
  ):
    | { status: 'ok'; lease: { release(): void } }
    | { status: 'over-budget'; requestedBytes: number; availableBytes: number };
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

  // Each adjusted layer needs both a temporary surface and a same-sized
  // ImageData buffer in addition to the final composite surface.
  const surfaceCount = 1 + entry.layers.filter((layer) => layer.adjustments !== undefined).length * 2;
  const reservation = deps.reserve?.(rect.width * rect.height * 4 * surfaceCount);
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
      if (!deps.isSnapshotCurrent(snapshot)) {
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
