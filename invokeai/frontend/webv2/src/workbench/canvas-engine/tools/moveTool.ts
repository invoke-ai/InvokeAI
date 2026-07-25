/**
 * The move tool: drag to move the SELECTED layer.
 *
 * Interaction contract:
 * - **The layers panel is the sole authority on which layer is active.** The move
 *   tool never hit-tests the stack to pick a target and never dispatches
 *   `setCanvasSelectedLayer`. Clicking canvas pixels that belong to some other
 *   layer does not steal the selection, and clicking empty space does not clear
 *   it — an editor where the pointer silently re-targets is the behaviour this
 *   contract exists to prevent.
 * - **Click** (press+release under the drag threshold): a no-op.
 * - **Drag**: moves the document's `selectedLayerId`, when that layer is enabled
 *   and unlocked. Hidden/locked layers are never dragged, and a press over empty
 *   space still drags the selected layer (the grab point is irrelevant).
 *   `shift` constrains motion to the dominant axis. Pointer-move only sets a
 *   transient transform override (live preview) — it never dispatches.
 * - **Commit** (pointer-up after a real move): one `commitStructural` with the
 *   new/old transform x/y. A zero-delta drag commits nothing.
 * - **Cancel** (Esc / pointercancel): drops the override, no dispatch.
 *
 * Zero React, zero import-time side effects.
 */

import type { CanvasLayerContract } from '@workbench/canvas-engine/contracts';
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

interface GestureState {
  startDoc: Vec2;
  startScreen: Vec2;
  /** The layer being dragged (null when the selected layer isn't movable). */
  targetId: string | null;
  /** The drag target's original transform x/y. */
  origin: { x: number; y: number } | null;
  moved: boolean;
}

/** Creates a fresh move tool with its own gesture state. */
export const createMoveTool = (): Tool => {
  let state: GestureState | null = null;

  const clearOverride = (ctx: ToolContext): void => {
    if (state?.targetId) {
      ctx.setLayerTransformOverride(state.targetId, null);
    }
  };

  const endGesture = (): void => {
    state = null;
  };

  /**
   * The drag target is the document's selected layer, and nothing else — the
   * press point plays no part in choosing it.
   */
  const resolveDragTarget = (ctx: ToolContext): CanvasLayerContract | null => {
    const doc = ctx.getDocument();
    const selectedId = doc?.selectedLayerId;
    const selected = doc && selectedId ? doc.layers.find((layer) => layer.id === selectedId) : undefined;
    return selected && isDraggable(selected) ? selected : null;
  };

  const previewAt = (ctx: ToolContext, input: PointerInput): void => {
    if (!state || !state.targetId || !state.origin) {
      return;
    }
    const delta = constrainDelta(
      input.documentPoint.x - state.startDoc.x,
      input.documentPoint.y - state.startDoc.y,
      input.modifiers.shift
    );
    ctx.setLayerTransformOverride(state.targetId, { x: state.origin.x + delta.x, y: state.origin.y + delta.y });
  };

  return {
    cursor: () => 'move',
    id: 'move',
    onDeactivate: (ctx) => {
      clearOverride(ctx);
      endGesture();
    },
    onPointerCancel: (ctx) => {
      clearOverride(ctx);
      ctx.invalidate({ overlay: true });
      endGesture();
    },
    onPointerDown: (ctx, input) => {
      if (state || (input.buttons & PRIMARY_BUTTON) === 0) {
        return;
      }
      const doc = ctx.getDocument();
      if (!doc) {
        return;
      }
      const dragTarget = resolveDragTarget(ctx);
      state = {
        moved: false,
        origin: dragTarget ? { x: dragTarget.transform.x, y: dragTarget.transform.y } : null,
        startDoc: input.documentPoint,
        startScreen: input.screenPoint,
        targetId: dragTarget?.id ?? null,
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
      previewAt(ctx, input);
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

      if (!current.targetId || !current.origin) {
        // No movable layer is selected — nothing to commit.
        return;
      }

      const delta = constrainDelta(
        input.documentPoint.x - current.startDoc.x,
        input.documentPoint.y - current.startDoc.y,
        input.modifiers.shift
      );
      const next = { x: current.origin.x + delta.x, y: current.origin.y + delta.y };

      if (next.x === current.origin.x && next.y === current.origin.y) {
        // Zero-delta drag: drop the preview, commit nothing.
        ctx.setLayerTransformOverride(current.targetId, null);
        return;
      }

      ctx.commitStructural(
        'Move layer',
        { id: current.targetId, patch: { transform: { x: next.x, y: next.y } }, type: 'updateCanvasLayer' },
        {
          id: current.targetId,
          patch: { transform: { x: current.origin.x, y: current.origin.y } },
          type: 'updateCanvasLayer',
        }
      );
      // The committed transform now flows through the mirror; drop the preview.
      ctx.setLayerTransformOverride(current.targetId, null);
    },
  };
};
