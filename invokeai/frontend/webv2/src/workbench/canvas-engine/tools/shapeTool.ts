/**
 * The shape tool: drag on the canvas to CREATE a new shape layer sized by the
 * drag rect. Scope is creation-only — dragging never edits an existing shape
 * (param edits happen through the options bar and the transform tool).
 *
 * Interaction contract (CANVAS_PLAN Phase 6.1):
 * - **Pointer-down** (primary button) starts a gesture at the press point.
 * - **Pointer-move** (past a small threshold) updates a transient overlay
 *   preview rect (`stores.shapePreview`) — it never dispatches. Hold **shift**
 *   to constrain to a square/circle.
 * - **Commit** (pointer-up after a real drag): exactly one `commitStructural`
 *   whose forward is `addCanvasLayer` (a shape source sized to the rect, placed
 *   at the rect origin) and whose inverse is `removeCanvasLayers` — so undo
 *   removes the created layer. A zero-area drag commits nothing.
 * - **Cancel** (Esc / pointercancel): drops the preview, no dispatch.
 *
 * The selection mask does NOT constrain shape creation — a shape layer is a
 * parametric object, not a pixel edit into an existing layer.
 *
 * Zero React, zero import-time side effects.
 */

import type { Rect, Vec2 } from '@workbench/canvas-engine/types';
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';
import type { CanvasRasterLayerContractV2 } from '@workbench/types';

import type { Tool, ToolContext } from './tool';

/** Bit for the primary (usually left) mouse button in `PointerEvent.buttons`. */
const PRIMARY_BUTTON = 1;

/** Screen-space distance (CSS px) the pointer must travel before a press becomes a drag. */
export const SHAPE_DRAG_THRESHOLD_PX = 3;

interface GestureState {
  startDoc: Vec2;
  startScreen: Vec2;
  moved: boolean;
}

/** The integer, normalized document rect for a drag from `start` to `end`, optionally square-constrained. */
export const rectFromDrag = (start: Vec2, end: Vec2, square: boolean): Rect => {
  let dx = end.x - start.x;
  let dy = end.y - start.y;
  if (square) {
    const side = Math.max(Math.abs(dx), Math.abs(dy));
    dx = (dx < 0 ? -1 : 1) * side;
    dy = (dy < 0 ? -1 : 1) * side;
  }
  const x = Math.round(Math.min(start.x, start.x + dx));
  const y = Math.round(Math.min(start.y, start.y + dy));
  return { height: Math.round(Math.abs(dy)), width: Math.round(Math.abs(dx)), x, y };
};

/** Creates a fresh shape tool with its own gesture state. */
export const createShapeTool = (): Tool => {
  let state: GestureState | null = null;

  const clearPreview = (ctx: ToolContext): void => {
    ctx.stores.shapePreview.set(null);
    ctx.invalidate({ overlay: true });
  };

  return {
    cursor: () => 'crosshair',
    id: 'shape',
    onDeactivate: (ctx) => {
      state = null;
      clearPreview(ctx);
    },
    onKeyCommand: (ctx, command) => {
      if (command === 'cancel' && state) {
        state = null;
        clearPreview(ctx);
      }
    },
    onPointerCancel: (ctx) => {
      state = null;
      clearPreview(ctx);
    },
    onPointerDown: (ctx, input) => {
      if (state || (input.buttons & PRIMARY_BUTTON) === 0) {
        return;
      }
      if (!ctx.getDocument()) {
        return;
      }
      state = { moved: false, startDoc: input.documentPoint, startScreen: input.screenPoint };
    },
    onPointerMove: (ctx, input) => {
      if (!state) {
        return;
      }
      if (!state.moved) {
        const dxs = input.screenPoint.x - state.startScreen.x;
        const dys = input.screenPoint.y - state.startScreen.y;
        if (Math.hypot(dxs, dys) < SHAPE_DRAG_THRESHOLD_PX) {
          return;
        }
        state.moved = true;
      }
      const rect = rectFromDrag(state.startDoc, input.documentPoint, input.modifiers.shift);
      ctx.stores.shapePreview.set({ kind: ctx.stores.shapeOptions.get().kind, rect });
      ctx.invalidate({ overlay: true });
    },
    onPointerUp: (ctx, input) => {
      if (!state) {
        return;
      }
      const current = state;
      state = null;

      if (!current.moved) {
        clearPreview(ctx);
        return;
      }

      const rect = rectFromDrag(current.startDoc, input.documentPoint, input.modifiers.shift);
      if (rect.width < 1 || rect.height < 1) {
        // Degenerate drag: nothing to create.
        clearPreview(ctx);
        return;
      }

      const doc = ctx.getDocument();
      if (!doc) {
        clearPreview(ctx);
        return;
      }

      const options = ctx.stores.shapeOptions.get();
      const layerId = ctx.createLayerId();
      const layer: CanvasRasterLayerContractV2 = {
        blendMode: 'normal',
        id: layerId,
        isEnabled: true,
        isLocked: false,
        name: `Shape ${doc.layers.length + 1}`,
        opacity: 1,
        source: {
          fill: options.fill,
          height: rect.height,
          kind: options.kind,
          stroke: options.stroke,
          strokeWidth: options.strokeWidth,
          type: 'shape',
          width: rect.width,
        },
        transform: { rotation: 0, scaleX: 1, scaleY: 1, x: rect.x, y: rect.y },
        type: 'raster',
      };

      const forward: CanvasProjectMutation = { index: 0, layer, type: 'addCanvasLayer' };
      const inverse: CanvasProjectMutation = { ids: [layerId], type: 'removeCanvasLayers' };
      ctx.commitStructural('Add shape', forward, inverse);
      clearPreview(ctx);
    },
  };
};
