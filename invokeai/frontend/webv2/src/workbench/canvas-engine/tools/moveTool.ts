/**
 * The move tool: click to select the top-most layer under the cursor, drag to
 * move a layer (auto-selecting the pressed unlocked layer, photo-editor style).
 *
 * Interaction contract (see CANVAS_PLAN Phase 3):
 * - **Click** (press+release under the drag threshold): selects the top-most
 *   VISIBLE layer whose rendered bounds contain the point (locked layers may be
 *   click-selected); empty space clears the selection. One `setCanvasSelectedLayer`
 *   dispatch, no history entry.
 * - **Drag**: moves a layer. The pressed point's top-most enabled+unlocked layer
 *   becomes the target (auto-select); otherwise the currently-selected
 *   enabled+unlocked layer is moved. Hidden/locked layers are never dragged.
 *   `shift` constrains motion to the dominant axis. Pointer-move only sets a
 *   transient transform override (live preview) — it never dispatches.
 * - **Commit** (pointer-up after a real move): one `commitStructural` with the
 *   new/old transform x/y (plus a selection dispatch when the target wasn't
 *   already selected). A zero-delta drag commits nothing.
 * - **Cancel** (Esc / pointercancel): drops the override, no dispatch.
 *
 * Zero React, zero import-time side effects.
 */

import type { PointerInput, Vec2 } from '@workbench/canvas-engine/types';
import type { CanvasLayerContract } from '@workbench/types';

import type { Tool, ToolContext } from './tool';

import { type LiveCacheRectOf, topLayerAt } from './moveHitTest';

/** Bit for the primary (usually left) mouse button in `PointerEvent.buttons`. */
const PRIMARY_BUTTON = 1;

/**
 * A live-cache-rect resolver over the tool's layer cache, so hit-testing grabs
 * freshly-painted content the debounced bitmap flush hasn't yet written back to
 * the persisted contract (a new layer + stroke is otherwise ungrabbable until the
 * ~1.5s flush lands).
 */
const liveRectOf =
  (ctx: ToolContext): LiveCacheRectOf =>
  (layerId) =>
    ctx.layers.get(layerId)?.rect;

/** Screen-space distance (CSS px) the pointer must travel before a press becomes a drag. */
export const MOVE_DRAG_THRESHOLD_PX = 3;

const isVisible = (layer: CanvasLayerContract): boolean => layer.isEnabled;
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
  /** The layer being dragged (null when the press had nothing to move). */
  targetId: string | null;
  /** The drag target's original transform x/y. */
  origin: { x: number; y: number } | null;
  /** The top-most visible layer under the press, for click selection (lock allowed). */
  clickTargetId: string | null;
  /** The document's selected layer at press time. */
  selectedAtStart: string | null;
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

  const resolveDragTarget = (ctx: ToolContext, point: Vec2, selectedId: string | null): CanvasLayerContract | null => {
    const doc = ctx.getDocument();
    if (!doc) {
      return null;
    }
    // Auto-select: the pressed point's top-most enabled+unlocked layer wins.
    const hit = topLayerAt(doc, point, isDraggable, liveRectOf(ctx));
    if (hit) {
      return hit;
    }
    // Otherwise fall back to moving the currently-selected layer (if movable).
    const selected = selectedId ? doc.layers.find((layer) => layer.id === selectedId) : undefined;
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
      const clickTarget = topLayerAt(doc, input.documentPoint, isVisible, liveRectOf(ctx));
      const dragTarget = resolveDragTarget(ctx, input.documentPoint, doc.selectedLayerId);
      state = {
        clickTargetId: clickTarget?.id ?? null,
        moved: false,
        origin: dragTarget ? { x: dragTarget.transform.x, y: dragTarget.transform.y } : null,
        selectedAtStart: doc.selectedLayerId,
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
        // A click: select the top-most visible layer, or clear on empty space.
        ctx.dispatch({ id: current.clickTargetId, type: 'setCanvasSelectedLayer' });
        return;
      }

      if (!current.targetId || !current.origin) {
        // Dragged over empty space with nothing to move — no-op.
        return;
      }

      const delta = constrainDelta(
        input.documentPoint.x - current.startDoc.x,
        input.documentPoint.y - current.startDoc.y,
        input.modifiers.shift
      );
      const next = { x: current.origin.x + delta.x, y: current.origin.y + delta.y };

      if (next.x === current.origin.x && next.y === current.origin.y) {
        // Zero-delta drag: behave like a click-select of the target, no commit.
        ctx.setLayerTransformOverride(current.targetId, null);
        ctx.dispatch({ id: current.targetId, type: 'setCanvasSelectedLayer' });
        return;
      }

      // Auto-select the dragged layer if it wasn't already selected (no history).
      if (current.targetId !== current.selectedAtStart) {
        ctx.dispatch({ id: current.targetId, type: 'setCanvasSelectedLayer' });
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
