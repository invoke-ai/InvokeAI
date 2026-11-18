import type { CanvasDocumentContractV3, Rect, StructuralCommitResult } from '@workbench/canvas-engine/api';
import type { CanvasGallerySaveRegion } from '@workbench/canvas-operations/api';

/** Use the header's structural engine subset to keep command tests React-free. */
export interface CanvasHeaderCommandEngine {
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

/** What the header's commands read and act through. */
export interface CanvasHeaderCommandContext {
  readonly engine: CanvasHeaderCommandEngine;
  readonly document: CanvasDocumentContractV3;
  /** An in-flight operation owns the document; every mutating header action is inert. */
  readonly editingLocked: boolean;
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

/** The header's commands as `[id, titleKey, defaultKeys]`; the header registers them while the Canvas is mounted. */
export const CANVAS_HEADER_HOTKEYS = [
  ['canvas.fitBboxToLayers', 'widgets.canvas.controls.fitBboxToLayers', ['shift+n']],
  ['canvas.fitBboxToMasks', 'widgets.canvas.controls.fitBboxToMasks', ['shift+b']],
  // The gallery saves have no default keys; they are assignable through the hotkeys settings.
  ['canvas.saveToGallery', 'widgets.canvas.contextMenu.saveCanvasToGallery', []],
  ['canvas.saveBboxToGallery', 'widgets.canvas.contextMenu.saveBboxToGallery', []],
  // Clear canvas keeps its saved-binding id, `newSession`.
  ['canvas.newSession', 'widgets.canvas.controls.clearCanvas', ['mod+alt+n']],
  // Browsers reserve Mod+1…8 for tab switching; these act only where the browser delivers the chord.
  ['canvas.setZoomTo100Percent', 'widgets.canvas.commands.zoomTo100', ['mod+1']],
  ['canvas.setZoomTo200Percent', 'widgets.canvas.commands.zoomTo200', ['mod+2']],
  ['canvas.setZoomTo400Percent', 'widgets.canvas.commands.zoomTo400', ['mod+3']],
  ['canvas.setZoomTo800Percent', 'widgets.canvas.commands.zoomTo800', ['mod+4']],
] as const satisfies readonly (readonly [id: string, titleKey: string, defaultKeys: readonly string[]])[];

/** The zoom preset commands and the scale each sets about the viewport centre. */
const ZOOM_PRESET_COMMANDS: Readonly<Record<string, number>> = {
  'canvas.setZoomTo100Percent': 1,
  'canvas.setZoomTo200Percent': 2,
  'canvas.setZoomTo400Percent': 4,
  'canvas.setZoomTo800Percent': 8,
};

/** Route new-session commands through the button's confirmation dialog. */
export const executeCanvasHeaderCommand = (commandId: string, ctx: CanvasHeaderCommandContext): void => {
  const zoomPreset = ZOOM_PRESET_COMMANDS[commandId];
  if (zoomPreset !== undefined) {
    zoomAtViewportCentre(ctx.engine, zoomPreset);
  } else if (commandId === 'canvas.fitBboxToLayers') {
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
