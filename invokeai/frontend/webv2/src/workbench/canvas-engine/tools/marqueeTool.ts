/**
 * The marquee tool: rectangular and elliptical pixel-selection by drag.
 *
 * One tool covers both shapes, switched by `marqueeOptions.kind` — they share
 * every part of the gesture, and splitting them would double the entry in the
 * `ToolId` union, the tool strip, and the options bar for one differing line.
 * This mirrors `ShapeToolOptions.kind`.
 *
 * Interaction contract:
 * - **Pointer-down** (primary button) anchors the drag.
 * - **Pointer-move** (past a small threshold) publishes a live outline to
 *   `stores.marqueePreview` — it never dispatches and never touches the mask.
 *   **shift** constrains to a square/circle; **alt** draws from the press point
 *   as the CENTRE rather than a corner. Both compose.
 * - **Commit** (pointer-up after a real drag): one `ctx.commitSelection` with
 *   the shape's closed path and the boolean op resolved from the modifiers, or
 *   the persistent op mode when none are held — the same resolution the lasso
 *   uses (`selectionOpFor`), so the whole mask/boolean pipeline is reused
 *   unchanged. A degenerate (sub-pixel) drag commits nothing.
 * - **Cancel** (Esc / pointercancel): drops the preview, no commit.
 *
 * Note the modifier overlap: shift/alt pick the boolean op AND shape the rect.
 * That is deliberate and matches Photoshop — alt-dragging a marquee subtracts a
 * region drawn from its centre.
 *
 * Selection edits are transient interaction state: not dispatches, not undoable.
 * Zero React, zero import-time side effects.
 */

import type { Rect, Vec2 } from '@workbench/canvas-engine/types';

import { selectionOpFor } from '@workbench/canvas-engine/selection/selectionOpMode';
import { selectionShapePathData } from '@workbench/canvas-engine/selection/selectionPaths';

import type { Tool, ToolContext } from './tool';

/** Bit for the primary (usually left) mouse button in `PointerEvent.buttons`. */
const PRIMARY_BUTTON = 1;

/** Screen-space distance (CSS px) the pointer must travel before a press becomes a drag. */
export const MARQUEE_DRAG_THRESHOLD_PX = 3;

interface GestureState {
  startDoc: Vec2;
  startScreen: Vec2;
  moved: boolean;
}

/** How the drag modifiers shape the rect. */
export interface MarqueeConstraints {
  /** Equal width and height (shift). */
  square: boolean;
  /** The press point is the rect's centre rather than a corner (alt). */
  fromCenter: boolean;
}

/** The drag delta, optionally forced to equal magnitude on both axes (sign kept). */
const constrainedDelta = (start: Vec2, end: Vec2, square: boolean): Vec2 => {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  if (!square) {
    return { x: dx, y: dy };
  }
  const side = Math.max(Math.abs(dx), Math.abs(dy));
  return { x: (dx < 0 ? -1 : 1) * side, y: (dy < 0 ? -1 : 1) * side };
};

/**
 * The integer, normalized document rect for a marquee drag from `start` to
 * `end`. With `fromCenter` the press point is the centre, so the constrained
 * delta is mirrored through it — giving a rect twice the drag extent.
 */
export const marqueeRect = (start: Vec2, end: Vec2, constraints: MarqueeConstraints): Rect => {
  const delta = constrainedDelta(start, end, constraints.square);
  const a = constraints.fromCenter ? { x: start.x - delta.x, y: start.y - delta.y } : start;
  const b = { x: start.x + delta.x, y: start.y + delta.y };
  const left = Math.round(Math.min(a.x, b.x));
  const top = Math.round(Math.min(a.y, b.y));
  return {
    height: Math.round(Math.max(a.y, b.y)) - top,
    width: Math.round(Math.max(a.x, b.x)) - left,
    x: left,
    y: top,
  };
};

/** Reads the shaping constraints out of a pointer sample's modifiers. */
const constraintsFor = (modifiers: { shift: boolean; alt: boolean }): MarqueeConstraints => ({
  fromCenter: modifiers.alt,
  square: modifiers.shift,
});

/** Creates a fresh marquee tool with its own gesture state. */
export const createMarqueeTool = (): Tool => {
  let state: GestureState | null = null;

  const clearPreview = (ctx: ToolContext): void => {
    ctx.stores.marqueePreview.set(null);
    ctx.invalidate({ overlay: true });
  };

  const end = (): void => {
    state = null;
  };

  return {
    cursor: () => 'crosshair',
    id: 'marquee',
    onDeactivate: (ctx) => {
      end();
      clearPreview(ctx);
    },
    onKeyCommand: (ctx, command) => {
      if (command === 'cancel' && state) {
        end();
        clearPreview(ctx);
      }
    },
    onPointerCancel: (ctx) => {
      end();
      clearPreview(ctx);
    },
    onPointerDown: (ctx, input) => {
      if (state || (input.buttons & PRIMARY_BUTTON) === 0 || !ctx.getDocument()) {
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
        if (Math.hypot(dxs, dys) < MARQUEE_DRAG_THRESHOLD_PX) {
          return;
        }
        state.moved = true;
      }
      const rect = marqueeRect(state.startDoc, input.documentPoint, constraintsFor(input.modifiers));
      ctx.stores.marqueePreview.set({ kind: ctx.stores.marqueeOptions.get().kind, rect });
      ctx.invalidate({ overlay: true });
    },
    onPointerUp: (ctx, input) => {
      if (!state) {
        return;
      }
      const current = state;
      end();
      clearPreview(ctx);

      if (!current.moved || !ctx.commitSelection) {
        return;
      }
      const rect = marqueeRect(current.startDoc, input.documentPoint, constraintsFor(input.modifiers));
      if (rect.width < 1 || rect.height < 1) {
        // Degenerate drag: selecting nothing would silently wipe the selection
        // under `replace`, so treat it as a no-op instead.
        return;
      }
      const { kind } = ctx.stores.marqueeOptions.get();
      const op = selectionOpFor(input.modifiers, ctx.stores.marqueeOptions.get().mode);
      ctx.commitSelection({
        bounds: rect,
        op,
        path: ctx.createPath2D(selectionShapePathData(kind, rect)),
      });
    },
  };
};
