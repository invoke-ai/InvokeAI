/**
 * Bbox drags preview without dispatch. Handles resize, interior moves and outside is inert. Alt bypasses grid
 * while retaining integer pixels; Shift constrains movement/aspect, Ctrl/Command resizes symmetrically. Read
 * modifiers per sample. Release commits one changed bbox with undo; cancellation and zero change commit nothing.
 */

import type { Rect, Vec2 } from '@workbench/canvas-engine/types';

import { constrainMoveDelta } from '@workbench/canvas-engine/transform/transformMath';

import type { Tool, ToolContext } from './tool';

import { type BboxTarget, bboxEquals, bboxTargetAt, moveBbox, resizeBbox, roundBbox } from './bboxHitTest';

/** Bit for the primary (usually left) mouse button in `PointerEvent.buttons`. */
const PRIMARY_BUTTON = 1;

const CURSOR_FOR_TARGET: Record<BboxTarget, string> = {
  e: 'ew-resize',
  move: 'move',
  n: 'ns-resize',
  ne: 'nesw-resize',
  nw: 'nwse-resize',
  s: 'ns-resize',
  se: 'nwse-resize',
  sw: 'nesw-resize',
  w: 'ew-resize',
};

/** Screen-space distance (CSS px) the pointer must travel before a press becomes a drag. */
export const BBOX_DRAG_THRESHOLD_PX = 3;

interface GestureState {
  target: BboxTarget;
  /** The bbox at gesture start (document space). */
  startBbox: Rect;
  /** The pointer's document/screen position at gesture start. */
  startDoc: Vec2;
  startScreen: Vec2;
  moved: boolean;
}

/** The bbox projected to screen space (axis-aligned; the view has no rotation). */
const bboxToScreen = (ctx: ToolContext, bbox: Rect): Rect => {
  const topLeft = ctx.viewport.documentToScreen({ x: bbox.x, y: bbox.y });
  const bottomRight = ctx.viewport.documentToScreen({ x: bbox.x + bbox.width, y: bbox.y + bbox.height });
  return {
    height: bottomRight.y - topLeft.y,
    width: bottomRight.x - topLeft.x,
    x: topLeft.x,
    y: topLeft.y,
  };
};

/** Computes the next bbox for a gesture from the current pointer, applying snap/aspect rules. */
const nextBboxFor = (
  ctx: ToolContext,
  state: GestureState,
  point: Vec2,
  modifiers: { alt: boolean; ctrl: boolean; meta: boolean; shift: boolean }
): Rect => {
  const grid = ctx.stores.bboxGrid.get();
  // Snap to the model grid unless Alt bypasses it or the snap-to-grid setting is off.
  // The geometry helpers still apply baseline whole-pixel alignment in either bypass case.
  const snap = !modifiers.alt && ctx.stores.snapToGrid.get();
  const dx = point.x - state.startDoc.x;
  const dy = point.y - state.startDoc.y;

  if (state.target === 'move') {
    const delta = constrainMoveDelta({ x: dx, y: dy }, modifiers.shift);
    return moveBbox(state.startBbox, delta.x, delta.y, grid, snap);
  }

  const options = ctx.stores.bboxOptions.get();
  const constrain = options.aspectLocked || modifiers.shift;
  // Locked → the option ratio; unlocked+shift → preserve the frame's own ratio.
  const ratio = options.aspectLocked
    ? options.aspectRatio
    : state.startBbox.height > 0
      ? state.startBbox.width / state.startBbox.height
      : 1;

  // Ctrl (⌘ on macOS) mirrors the drag across the frame's center.
  const symmetric = modifiers.ctrl || modifiers.meta;

  return resizeBbox({
    constrain,
    dx,
    dy,
    grid,
    handle: state.target,
    ratio,
    snap,
    start: state.startBbox,
    symmetric,
  });
};

/** Creates a fresh bbox tool with its own gesture state. */
export const createBboxTool = (): Tool => {
  let state: GestureState | null = null;
  let hoverTarget: BboxTarget | null = null;

  const clearPreview = (ctx: ToolContext): void => {
    ctx.stores.bboxPreview.set(null);
    ctx.invalidate({ overlay: true });
  };

  return {
    // Hold the grabbed cursor during drag; otherwise use hovered handle/interior or default arrow.
    cursor: () => {
      const target = state?.target ?? hoverTarget;
      return target ? CURSOR_FOR_TARGET[target] : 'default';
    },
    id: 'bbox',
    onDeactivate: (ctx) => {
      state = null;
      hoverTarget = null;
      clearPreview(ctx);
    },
    onPointerCancel: (ctx) => {
      state = null;
      clearPreview(ctx);
    },
    onPointerDown: (ctx, input) => {
      if (state || (input.buttons & PRIMARY_BUTTON) === 0) {
        return;
      }
      const doc = ctx.getDocument();
      if (!doc) {
        return;
      }
      const target = bboxTargetAt(bboxToScreen(ctx, doc.bbox), input.screenPoint);
      if (!target) {
        // Outside the frame: the bbox is never deselected, so do nothing.
        return;
      }
      state = {
        moved: false,
        startBbox: doc.bbox,
        startDoc: input.documentPoint,
        startScreen: input.screenPoint,
        target,
      };
    },
    onPointerMove: (ctx, input) => {
      if (!state) {
        // Idle hover: reflect the handle/interior under the pointer in the cursor.
        const doc = ctx.getDocument();
        const target = doc ? bboxTargetAt(bboxToScreen(ctx, doc.bbox), input.screenPoint) : null;
        if (target !== hoverTarget) {
          hoverTarget = target;
          ctx.updateCursor();
        }
        return;
      }
      if (!state.moved) {
        const dxs = input.screenPoint.x - state.startScreen.x;
        const dys = input.screenPoint.y - state.startScreen.y;
        if (Math.hypot(dxs, dys) < BBOX_DRAG_THRESHOLD_PX) {
          return;
        }
        state.moved = true;
      }
      const next = nextBboxFor(ctx, state, input.documentPoint, input.modifiers);
      ctx.stores.bboxPreview.set(next);
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

      const next = roundBbox(nextBboxFor(ctx, current, input.documentPoint, input.modifiers));
      if (bboxEquals(next, current.startBbox)) {
        // Zero-delta gesture: nothing to commit.
        clearPreview(ctx);
        return;
      }

      ctx.commitStructural(
        'Set generation frame',
        { bbox: next, type: 'setCanvasBbox' },
        { bbox: roundBbox(current.startBbox), type: 'setCanvasBbox' }
      );
      // The committed bbox now flows through the mirror; drop the preview.
      clearPreview(ctx);
    },
  };
};
