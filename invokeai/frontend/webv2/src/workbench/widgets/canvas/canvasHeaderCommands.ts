import type {
  CanvasDocumentCapability,
  CanvasDocumentContractV3,
  Rect,
  StructuralCommitResult,
} from '@workbench/canvas-engine/api';
import type { CanvasGallerySaveRegion } from '@workbench/canvas-operations/api';

import { createNewCanvasState } from '@workbench/canvasMigration';

/** Use the header's structural engine subset to keep command tests React-free. */
export interface CanvasHeaderCommandEngine {
  readonly document: Pick<CanvasDocumentCapability, 'replaceDocument'>;
  readonly layers: {
    commitStructural(label: string, forward: unknown, inverse: unknown): StructuralCommitResult;
  };
  readonly viewport: {
    fitToView(): void;
    getViewport(): {
      getViewportSize(): { width: number; height: number };
      zoomAtPoint(newZoom: number, screenAnchor: { x: number; y: number }): void;
    };
  };
}

/** The minimum the destructive replace needs — kept narrow so its memo deps stay tight. */
export interface NewCanvasContext {
  readonly document: Pick<CanvasDocumentContractV3, 'width' | 'height'>;
  /** An in-flight operation owns the document; every mutating header action is inert. */
  readonly editingLocked: boolean;
  readonly engine: Pick<CanvasHeaderCommandEngine, 'document'>;
}

export interface CanvasHeaderCommandContext extends NewCanvasContext {
  readonly engine: CanvasHeaderCommandEngine;
  readonly document: CanvasDocumentContractV3;
  readonly fitLayersRect: Rect | null;
  readonly fitMasksRect: Rect | null;
  /** Opens the confirm dialog. The destructive replace only runs from its confirm. */
  readonly openNewCanvas: () => void;
  readonly reportStructuralCommit: (result: StructuralCommitResult) => void;
  /** Composites the region and uploads it to the gallery's current board. */
  readonly saveToGallery: (region: CanvasGallerySaveRegion) => void;
  readonly t: (key: string) => string;
}

/** Zooms about the viewport's centre, the anchor every header zoom control uses. */
export const zoomAtViewportCentre = (engine: CanvasHeaderCommandEngine, value: number): void => {
  const viewport = engine.viewport.getViewport();
  const size = viewport.getViewportSize();
  viewport.zoomAtPoint(value, { x: size.width / 2, y: size.height / 2 });
};

/** Commit one undoable bbox change; optionally refit afterward. Null bounds are a no-op. */
export const applyFitBbox = (ctx: CanvasHeaderCommandContext, rect: Rect | null, refit: boolean): void => {
  if (ctx.editingLocked || !rect) {
    return;
  }
  const result = ctx.engine.layers.commitStructural(
    ctx.t('widgets.canvas.commands.fitBbox'),
    { bbox: rect, type: 'setCanvasBbox' },
    { bbox: ctx.document.bbox, type: 'setCanvasBbox' }
  );
  ctx.reportStructuralCommit(result);
  if (refit && result.status === 'committed') {
    ctx.engine.viewport.fitToView();
  }
};

/**
 * Replaces the whole document (seeded with one empty inpaint mask) at the current
 * dimensions. The engine's mirror treats this as a document swap and clears the
 * canvas history by design, so this is intentionally NOT undoable — the confirm
 * dialog is the safety net, and the editing lock is the second gate.
 */
export const confirmNewCanvas = (ctx: NewCanvasContext): void => {
  if (ctx.editingLocked) {
    return;
  }
  ctx.engine.document.replaceDocument(createNewCanvasState(ctx.document.width, ctx.document.height).document);
};

/** Route new-session commands through the button's confirmation dialog. */
export const executeCanvasHeaderCommand = (commandId: string, ctx: CanvasHeaderCommandContext): void => {
  if (commandId === 'canvas.fitBboxToLayers') {
    applyFitBbox(ctx, ctx.fitLayersRect, true);
  } else if (commandId === 'canvas.fitBboxToMasks') {
    applyFitBbox(ctx, ctx.fitMasksRect, false);
  } else if (commandId === 'canvas.newSession') {
    ctx.openNewCanvas();
  } else if (commandId === 'canvas.saveToGallery' && !ctx.editingLocked) {
    ctx.saveToGallery('canvas');
  } else if (commandId === 'canvas.saveBboxToGallery' && !ctx.editingLocked) {
    ctx.saveToGallery('bbox');
  }
};
