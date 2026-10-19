import type {
  CanvasDocumentCapability,
  CanvasExportCapability,
  CanvasLifecycleCapability,
} from '@workbench/canvas-engine/api';
import type { Rect } from '@workbench/canvas-engine/types';

export type CanvasCompositeRegion = 'canvas' | 'bbox';

export type ExportCanvasCompositeResult =
  | { status: 'ok'; blob: Blob; rect: Rect }
  | { status: 'empty' | 'stale' | 'not-ready' | 'over-budget' };

export type CanvasCompositeExportEngine = {
  readonly document: CanvasDocumentCapability;
  readonly exports: Pick<CanvasExportCapability, 'exportRasterComposite'>;
  readonly lifecycle: Pick<CanvasLifecycleCapability, 'flushPendingUploads'>;
};

/**
 * Flushes in-flight paint uploads, then composites the visible raster layers
 * of either the whole canvas content or the post-flush generation bbox.
 */
export const exportCanvasComposite = async (
  engine: CanvasCompositeExportEngine,
  region: CanvasCompositeRegion
): Promise<ExportCanvasCompositeResult> => {
  await engine.lifecycle.flushPendingUploads();
  const document = engine.document.getDocument();
  if (!document) {
    return { status: 'not-ready' };
  }

  return engine.exports.exportRasterComposite(
    region === 'canvas' ? { bounds: 'content' } : { bounds: 'rect', rect: document.bbox }
  );
};
