/**
 * Rect/ellipse marquee previews do not change the mask. Shift constrains aspect; Alt draws from center. On release
 * those same modifiers choose the shared boolean op, intentionally composing shape and operation.
 * Degenerate/cancelled drags do not commit; engine selection history records accepted changes.
 */

import type { Vec2 } from '@workbench/canvas-engine/types';

import { selectionOpFor } from '@workbench/canvas-engine/selection/selectionOpMode';
import { selectionShapePathData } from '@workbench/canvas-engine/selection/selectionPaths';

import type { Tool, ToolContext } from './tool';

import { dragRect, type DragRectConstraints } from './dragRect';

/** Bit for the primary (usually left) mouse button in `PointerEvent.buttons`. */
const PRIMARY_BUTTON = 1;

/** Screen-space distance (CSS px) the pointer must travel before a press becomes a drag. */
export const MARQUEE_DRAG_THRESHOLD_PX = 3;

interface GestureState {
  startDoc: Vec2;
  startScreen: Vec2;
  moved: boolean;
  /**
   * Shift/alt held at the press pick the boolean op (add/subtract); the same
   * keys pressed mid-drag shape the rect instead (square / from centre).
   */
  opModifiers: { shift: boolean; alt: boolean };
  /** Op keys let go during the drag: pressing one again shapes like any mid-drag press. */
  released: { shift: boolean; alt: boolean };
}

/** The shaping constraints: a key still held since it chose the op at the press does not also shape. */
const constraintsFor = (modifiers: { shift: boolean; alt: boolean }, state: GestureState): DragRectConstraints => {
  if (state.opModifiers.shift && !modifiers.shift) {
    state.released.shift = true;
  }
  if (state.opModifiers.alt && !modifiers.alt) {
    state.released.alt = true;
  }
  return {
    fromCenter: modifiers.alt && (!state.opModifiers.alt || state.released.alt),
    square: modifiers.shift && (!state.opModifiers.shift || state.released.shift),
  };
};

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
    usesAltKey: true,
    onDeactivate: (ctx) => {
      end();
      clearPreview(ctx);
    },
    onKeyCommand: (ctx, command) => {
      if (command !== 'cancel' || !state) {
        return false;
      }
      end();
      clearPreview(ctx);
      return true;
    },
    onPointerCancel: (ctx) => {
      end();
      clearPreview(ctx);
    },
    onPointerDown: (ctx, input) => {
      if (state || (input.buttons & PRIMARY_BUTTON) === 0 || !ctx.getDocument()) {
        return;
      }
      state = {
        moved: false,
        opModifiers: { alt: input.modifiers.alt, shift: input.modifiers.shift },
        released: { alt: false, shift: false },
        startDoc: input.documentPoint,
        startScreen: input.screenPoint,
      };
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
      const rect = dragRect(state.startDoc, input.documentPoint, constraintsFor(input.modifiers, state));
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
      const rect = dragRect(current.startDoc, input.documentPoint, constraintsFor(input.modifiers, current));
      if (rect.width < 1 || rect.height < 1) {
        // Degenerate drag: selecting nothing would silently wipe the selection
        // under `replace`, so treat it as a no-op instead.
        return;
      }
      const { kind } = ctx.stores.marqueeOptions.get();
      const op = selectionOpFor(current.opModifiers, ctx.stores.marqueeOptions.get().mode);
      ctx.commitSelection({
        bounds: rect,
        op,
        path: ctx.createPath2D(selectionShapePathData(kind, rect)),
      });
    },
  };
};
