/**
 * The text tool: click to CREATE editable-forever text, or click an existing
 * text layer to re-edit it.
 *
 * Interaction contract (CANVAS_PLAN Phase 6.2):
 * - **Click on empty area** → open a CREATE-mode text-editing session at that
 *   document point (style seeded from the text options bar). Nothing is added to
 *   the document until the session commits (a single `addCanvasLayer`).
 * - **Click on an existing text layer** (top-most, enabled, unlocked) → open an
 *   EDIT-mode session on it. Hit-testing inverts the layer transform and checks
 *   the point against the layer's rendered text-block rect (cache size, or the
 *   pure estimate before a cache exists). Locked/hidden text layers are skipped.
 * - **Click while a session is open** → the pointer pipeline commits the open
 *   session engine-side (`maybeCommitModalSession` → `commitOpenTextSession`,
 *   reading the live portal content) and swallows the press BEFORE it reaches
 *   this tool, so `onPointerDown` below is never invoked for that press. A
 *   subsequent click then places/edits. The `textEditSession` guard here is a
 *   defensive backstop for a harness that routes the press through anyway.
 * - **Commit** is engine-side on a canvas pointerdown (above); the portal's blur
 *   (focus lost to non-canvas UI) and `mod+enter` also commit. The live typed
 *   text is read from the portal only at commit time (no per-keystroke traffic).
 * - **Escape** is handled by the focused contenteditable (cancels the session);
 *   a defocused-but-open session is cancelled by the engine's Escape chain
 *   (`handleEscape`: text → transform → deselect), not routed to this tool.
 * - A **real** tool switch cancels the session (`onDeactivate`); a **temporary**
 *   modifier-hold switch (space→view / alt→colorPicker) preserves it, mirroring
 *   the transform tool.
 *
 * Text layers are not hit-testable by the move/transform tools (like shapes and
 * gradients), so this tool owns its own text hit-test.
 *
 * Zero React, zero import-time side effects.
 */

import type {
  CanvasDocumentContractV2,
  CanvasLayerContract,
  CanvasLayerSourceContract,
} from '@workbench/canvas-engine/contracts';
import type { Vec2 } from '@workbench/canvas-engine/types';

import { applyToPoint, invert } from '@workbench/canvas-engine/math/mat2d';
import { estimateTextExtent } from '@workbench/canvas-engine/render/rasterizers/textRasterizer';

import type { Tool, ToolContext } from './tool';

import { layerMatrix } from './moveHitTest';

/** Bit for the primary (usually left) mouse button in `PointerEvent.buttons`. */
const PRIMARY_BUTTON = 1;

type TextSource = Extract<CanvasLayerSourceContract, { type: 'text' }>;
/** A text-sourced raster layer. */
type TextLayer = Extract<CanvasLayerContract, { type: 'raster' }> & { source: TextSource };

/** True when `layer` is an enabled, unlocked text layer (an edit-session candidate). */
const isEditableTextLayer = (layer: CanvasLayerContract): layer is TextLayer =>
  layer.type === 'raster' && layer.source.type === 'text' && layer.isEnabled && !layer.isLocked;

/**
 * The rendered text-block size for hit-testing: the live cache surface size when
 * one exists (the precise, measured extent), else the pure estimate (before the
 * layer has been rasterized once).
 */
const textLayerSize = (layer: TextLayer, ctx: ToolContext): { width: number; height: number } => {
  const cache = ctx.layers.get(layer.id);
  if (cache) {
    return { height: cache.surface.height, width: cache.surface.width };
  }
  return estimateTextExtent(layer.source);
};

/** The top-most editable text layer whose rendered block contains `point` (document space), or `null`. */
const topTextLayerAt = (doc: CanvasDocumentContractV2, point: Vec2, ctx: ToolContext): TextLayer | null => {
  for (const layer of doc.layers) {
    if (!isEditableTextLayer(layer)) {
      continue;
    }
    const inverse = invert(layerMatrix(layer.transform));
    if (!inverse) {
      continue;
    }
    const local = applyToPoint(inverse, point);
    const size = textLayerSize(layer, ctx);
    if (local.x >= 0 && local.x <= size.width && local.y >= 0 && local.y <= size.height) {
      return layer;
    }
  }
  return null;
};

/** Creates a fresh text tool. It holds no gesture state (a click opens a session and returns). */
export const createTextTool = (): Tool => ({
  cursor: () => 'text',
  id: 'text',
  onDeactivate: (ctx, opts) => {
    if (opts?.temporary) {
      // A modifier-hold switch (space/alt) preserves the open session for
      // `onActivate` to resume when the hold ends — like the transform tool.
      return;
    }
    // A real tool switch cancels the session (in practice the contenteditable's
    // blur has already committed by now; this is the safety teardown).
    ctx.cancelTextEdit?.();
  },
  onKeyCommand: (ctx, command) => {
    // Defensive backstop: the pipeline does not route 'cancel' to tools (the
    // engine's `handleEscape` chain owns text/transform/deselect), so this only
    // fires if a harness routes it directly. Cancel drops a defocused session;
    // apply is a no-op here (only the portal holds the live content to commit).
    if (command === 'cancel') {
      ctx.cancelTextEdit?.();
    }
  },
  onPointerDown: (ctx, input) => {
    if ((input.buttons & PRIMARY_BUTTON) === 0) {
      return;
    }
    // Backstop: when a session is open the pipeline commits+swallows the press
    // before it reaches this tool, so this branch is normally unreached. Guard
    // anyway so a harness that routes the press through never opens a 2nd session.
    if (ctx.stores.textEditSession.get()) {
      return;
    }
    const doc = ctx.getDocument();
    if (!doc) {
      return;
    }
    const hit = topTextLayerAt(doc, input.documentPoint, ctx);
    if (hit) {
      ctx.openTextEdit?.(hit.id);
    } else {
      ctx.openTextCreate?.(input.documentPoint);
    }
  },
});
