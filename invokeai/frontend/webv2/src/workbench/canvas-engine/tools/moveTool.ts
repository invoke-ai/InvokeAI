/**
 * The move tool: drag to move the SELECTED layer, or the selected PIXELS.
 *
 * Interaction contract:
 * - **The layers panel is the sole authority on which layer is active.** The move
 *   tool never hit-tests the stack to pick a target and never dispatches
 *   `setCanvasSelectedLayer`. Clicking canvas pixels that belong to some other
 *   layer does not steal the selection, and clicking empty space does not clear
 *   it — an editor where the pointer silently re-targets is the behaviour this
 *   contract exists to prevent.
 * - **Click** (press+release under the drag threshold): a no-op.
 * - **Drag inside a live selection** — or with a float already in flight —
 *   moves the selected PIXELS: the first such drag cuts them out of the selected
 *   layer into a floating selection that follows the pointer, leaving a hole.
 *   The float stays live across further drags until it is committed (Enter,
 *   deselect, tool switch — one undo entry for the whole move) or cancelled
 *   (Escape). This is the Photoshop feel: ants present and the press inside them
 *   moves pixels; otherwise the press moves the layer.
 * - **Drag anywhere else**: moves the document's `selectedLayerId`, when that
 *   layer is enabled and unlocked. Hidden/locked layers are never dragged, and a
 *   press over empty space still drags the selected layer (the grab point is
 *   irrelevant). Pointer-move only sets a transient transform override (live
 *   preview) — it never dispatches.
 * - **`shift`** constrains motion to the dominant axis, for both modes.
 * - **Commit** (pointer-up after a real layer move): one `commitStructural` with
 *   the new/old transform x/y. A zero-delta drag commits nothing. Moving a float
 *   commits nothing here — the float's own commit does that later.
 * - **Cancel** (Esc / pointercancel): drops the layer override or reverts the
 *   float to where this drag started. Neither dispatches.
 *
 * Zero React, zero import-time side effects.
 */

import type { CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { LayerTransform } from '@workbench/canvas-engine/transform/transformMath';
import type { PointerInput, Vec2 } from '@workbench/canvas-engine/types';

import type { Tool, ToolContext } from './tool';

/** Bit for the primary (usually left) mouse button in `PointerEvent.buttons`. */
const PRIMARY_BUTTON = 1;

/** Screen-space distance (CSS px) the pointer must travel before a press becomes a drag. */
export const MOVE_DRAG_THRESHOLD_PX = 3;

const isDraggable = (layer: CanvasLayerContract): boolean => layer.isEnabled && !layer.isLocked;

/** Applies the shift-to-dominant-axis constraint to a document-space delta. */
export const constrainDelta = (dx: number, dy: number, shift: boolean): Vec2 => {
  if (!shift) {
    return { x: dx, y: dy };
  }
  return Math.abs(dx) >= Math.abs(dy) ? { x: dx, y: 0 } : { x: 0, y: dy };
};

/** Which thing this gesture moves. */
type DragMode =
  | { kind: 'layer'; targetId: string; origin: { x: number; y: number } }
  | { kind: 'float'; layerId: string; origin: LayerTransform }
  | { kind: 'none' };

interface GestureState {
  startDoc: Vec2;
  startScreen: Vec2;
  mode: DragMode;
  moved: boolean;
}

/** Creates a fresh move tool with its own gesture state. */
export const createMoveTool = (): Tool => {
  let state: GestureState | null = null;

  const clearOverride = (ctx: ToolContext): void => {
    if (state?.mode.kind === 'layer') {
      ctx.setLayerTransformOverride(state.mode.targetId, null);
    }
  };

  const endGesture = (): void => {
    state = null;
  };

  /**
   * The drag target is the document's selected layer, and nothing else — the
   * press point plays no part in choosing it.
   */
  const selectedDraggableLayer = (ctx: ToolContext): CanvasLayerContract | null => {
    const doc = ctx.getDocument();
    const selectedId = doc?.selectedLayerId;
    const selected = doc && selectedId ? doc.layers.find((layer) => layer.id === selectedId) : undefined;
    return selected && isDraggable(selected) ? selected : null;
  };

  /**
   * Resolves what a press at `point` drags. Pixels win over the layer when a
   * float is already in flight, or when a live selection contains the press.
   */
  const resolveMode = (ctx: ToolContext, point: Vec2): DragMode => {
    const existing = ctx.getFloatingSelection?.() ?? null;
    if (existing) {
      return { kind: 'float', layerId: existing.layerId, origin: { ...existing.transform } };
    }
    const layer = selectedDraggableLayer(ctx);
    if (!layer) {
      return { kind: 'none' };
    }
    if (ctx.isPointInSelection?.(point) && ctx.liftFloatingSelection?.(layer.id)) {
      const lifted = ctx.getFloatingSelection?.();
      if (lifted) {
        return { kind: 'float', layerId: lifted.layerId, origin: { ...lifted.transform } };
      }
    }
    return { kind: 'layer', origin: { x: layer.transform.x, y: layer.transform.y }, targetId: layer.id };
  };

  /** The constrained document-space delta from the gesture start to `input`. */
  const deltaFor = (current: GestureState, input: PointerInput): Vec2 =>
    constrainDelta(
      input.documentPoint.x - current.startDoc.x,
      input.documentPoint.y - current.startDoc.y,
      input.modifiers.shift
    );

  const applyDelta = (ctx: ToolContext, current: GestureState, input: PointerInput): void => {
    const delta = deltaFor(current, input);
    if (current.mode.kind === 'layer') {
      ctx.setLayerTransformOverride(current.mode.targetId, {
        x: current.mode.origin.x + delta.x,
        y: current.mode.origin.y + delta.y,
      });
      return;
    }
    if (current.mode.kind === 'float') {
      // The float's transform is LAYER-LOCAL, so a document-space pointer delta
      // has to be mapped through the layer's inverse matrix first — otherwise a
      // rotated or scaled layer would drag the pixels off at an angle.
      const local = ctx.documentDeltaToLayerLocal?.(current.mode.layerId, delta) ?? delta;
      ctx.setFloatingTransform?.({
        ...current.mode.origin,
        x: current.mode.origin.x + local.x,
        y: current.mode.origin.y + local.y,
      });
    }
  };

  return {
    cursor: () => 'move',
    id: 'move',
    onDeactivate: (ctx) => {
      clearOverride(ctx);
      endGesture();
    },
    onKeyCommand: (ctx, command) => {
      // Enter banks a float in place. Escape's abandon runs on the engine's own
      // ladder (it must work whichever tool is active), so it is not handled here.
      if (command === 'apply' && !state) {
        ctx.commitFloatingSelection?.();
      }
    },
    onPointerCancel: (ctx) => {
      if (state?.mode.kind === 'float') {
        // Revert just this drag; the float itself survives (Escape's own
        // float-cancel is a separate, engine-level step).
        ctx.setFloatingTransform?.(state.mode.origin);
      }
      clearOverride(ctx);
      ctx.invalidate({ overlay: true });
      endGesture();
    },
    onPointerDown: (ctx, input) => {
      if (state || (input.buttons & PRIMARY_BUTTON) === 0 || !ctx.getDocument()) {
        return;
      }
      state = {
        mode: resolveMode(ctx, input.documentPoint),
        moved: false,
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
        if (Math.hypot(dxs, dys) < MOVE_DRAG_THRESHOLD_PX) {
          return;
        }
        state.moved = true;
      }
      applyDelta(ctx, state, input);
    },
    onPointerUp: (ctx, input) => {
      if (!state) {
        return;
      }
      const current = state;
      endGesture();

      if (!current.moved) {
        // A click never re-targets the layer selection — that is the panel's job.
        return;
      }
      if (current.mode.kind === 'none') {
        // No movable layer is selected — nothing to commit.
        return;
      }
      if (current.mode.kind === 'float') {
        // The float keeps the drag; it bakes later, as one entry for the whole
        // move however many drags it took.
        applyDelta(ctx, current, input);
        return;
      }

      const delta = deltaFor(current, input);
      const origin = current.mode.origin;
      const next = { x: origin.x + delta.x, y: origin.y + delta.y };

      if (next.x === origin.x && next.y === origin.y) {
        // Zero-delta drag: drop the preview, commit nothing.
        ctx.setLayerTransformOverride(current.mode.targetId, null);
        return;
      }

      ctx.commitStructural(
        'Move layer',
        { id: current.mode.targetId, patch: { transform: { x: next.x, y: next.y } }, type: 'updateCanvasLayer' },
        { id: current.mode.targetId, patch: { transform: { x: origin.x, y: origin.y } }, type: 'updateCanvasLayer' }
      );
      // The committed transform now flows through the mirror; drop the preview.
      ctx.setLayerTransformOverride(current.mode.targetId, null);
    },
  };
};
