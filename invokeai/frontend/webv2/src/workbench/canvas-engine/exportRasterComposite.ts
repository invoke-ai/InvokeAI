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
  | { status: 'empty' | 'stale' | 'not-ready' };

export interface RasterCompositeExportSnapshot {
  document: CanvasDocumentContractV2 | null;
  documentGeneration: number;
}

export interface ExportRasterCompositeDeps extends RenderRasterCompositeDeps {
  backend: RenderRasterCompositeDeps['backend'] & {
    encodeSurface(surface: RasterSurface, type?: string): Promise<Blob>;
  };
  captureSnapshot(): RasterCompositeExportSnapshot;
  isSnapshotCurrent(snapshot: RasterCompositeExportSnapshot): boolean;
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

  let surface: RasterSurface;
  try {
    surface = await renderRasterComposite(entry, deps);
  } catch (error) {
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
};
