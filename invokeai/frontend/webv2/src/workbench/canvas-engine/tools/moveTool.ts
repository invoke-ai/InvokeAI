/**
 * Move targets panel selection, never stack hit tests. Dragging inside selection, or an existing float, moves
 * pixels across multiple drags until commit/cancel; the pixels are cut only once a press becomes a drag, so a click
 * leaves the layer whole. Otherwise move eligible selected layers with a primary-layer fallback.
 *
 * Shift locks the dominant axis. Layer movement uses the visible model grid unless Alt bypasses; local-space
 * floats never snap. Preview only during drag; changed layer moves commit once, float history waits for its later
 * bake. Cancel restores the drag start; clicks/zero movement do nothing.
 */

import type { CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { LayerTransform } from '@workbench/canvas-engine/transform/transformMath';
import type { PointerInput, Vec2 } from '@workbench/canvas-engine/types';

import { compileDocumentLeaves } from '@workbench/canvas-engine/document-model/documentModel';
import { getDocumentIndex } from '@workbench/canvas-engine/document/documentIndex';
import { isLeafEditable } from '@workbench/canvas-engine/document/layerEligibility';
import { snapMovedPoint } from '@workbench/canvas-engine/math/snapping';

import type { Tool, ToolContext } from './tool';

import { positionGrid } from './gridSnap';

/** Bit for the primary (usually left) mouse button in `PointerEvent.buttons`. */
const PRIMARY_BUTTON = 1;

/** Screen-space distance (CSS px) the pointer must travel before a press becomes a drag. */
export const MOVE_DRAG_THRESHOLD_PX = 3;

/** Applies the shift-to-dominant-axis constraint to a document-space delta. */
export const constrainDelta = (dx: number, dy: number, shift: boolean): Vec2 => {
  if (!shift) {
    return { x: dx, y: dy };
  }
  return Math.abs(dx) >= Math.abs(dy) ? { x: dx, y: 0 } : { x: 0, y: dy };
};

type LayerDrag = {
  kind: 'layer';
  targets: readonly { id: string; origin: { x: number; y: number }; primary: boolean }[];
};

/** Which thing this gesture moves. `lift` cuts the selected pixels only once the press becomes a drag. */
type DragMode =
  | LayerDrag
  | { kind: 'lift'; layerId: string; fallback: LayerDrag }
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
      for (const target of state.mode.targets) {
        ctx.setLayerTransformOverride(target.id, null);
      }
    }
  };

  const endGesture = (): void => {
    state = null;
  };

  /** Target comes solely from document selection, regardless of press location. */
  const selectedDraggableLayers = (
    ctx: ToolContext
  ): { layers: readonly CanvasLayerContract[]; primaryId: string | null } => {
    const doc = ctx.getDocument();
    const selectedId = doc?.selectedLayerId;
    if (!doc || !selectedId) {
      return { layers: [], primaryId: null };
    }
    const requested = new Set(ctx.getSelectedLayerIds?.() ?? [selectedId]);
    requested.add(selectedId);
    const index = getDocumentIndex(doc);
    if ([...requested].some((id) => !index.byId.has(id))) {
      return { layers: [], primaryId: null };
    }
    // A selected group drags every leaf beneath it; a locked node anywhere in the set refuses the drag.
    const leaves = compileDocumentLeaves(doc).filter(
      (leaf) => requested.has(leaf.id) || leaf.parentIds.some((ancestor) => requested.has(ancestor))
    );
    if (leaves.length === 0 || !leaves.every(isLeafEditable)) {
      return { layers: [], primaryId: null };
    }
    const primary = leaves.find((leaf) => leaf.id === selectedId || leaf.parentIds.includes(selectedId));
    return { layers: leaves.map((leaf) => leaf.layer), primaryId: primary?.id ?? null };
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
    const { layers, primaryId } = selectedDraggableLayers(ctx);
    const primary = layers.find((layer) => layer.id === primaryId);
    if (!primary) {
      return { kind: 'none' };
    }
    const layerDrag: LayerDrag = {
      kind: 'layer',
      targets: layers.map((layer) => ({
        id: layer.id,
        origin: { x: layer.transform.x, y: layer.transform.y },
        primary: layer === primary,
      })),
    };
    return ctx.isPointInSelection?.(point) && ctx.liftFloatingSelection
      ? { fallback: layerDrag, kind: 'lift', layerId: primary.id }
      : layerDrag;
  };

  /**
   * A click must leave the layer whole, so the pixels are cut only when the drag starts. Only a selection with
   * nothing to lift falls back to moving the layer; a refused lift moves nothing.
   */
  const liftOnDrag = (ctx: ToolContext, mode: Extract<DragMode, { kind: 'lift' }>): DragMode => {
    const result = ctx.liftFloatingSelection?.(mode.layerId) ?? 'unavailable';
    if (result === 'unavailable') {
      return mode.fallback;
    }
    const lifted = result === 'lifted' ? ctx.getFloatingSelection?.() : null;
    return lifted ? { kind: 'float', layerId: lifted.layerId, origin: { ...lifted.transform } } : { kind: 'none' };
  };

  /** The constrained document-space delta from the gesture start to `input`. */
  const deltaFor = (current: GestureState, input: PointerInput): Vec2 =>
    constrainDelta(
      input.documentPoint.x - current.startDoc.x,
      input.documentPoint.y - current.startDoc.y,
      input.modifiers.shift
    );

  /** Share snapped layer-origin calculation between preview and commit; Alt or settings may bypass. */
  const nextLayerPosition = (
    ctx: ToolContext,
    origin: { x: number; y: number },
    input: PointerInput,
    delta: Vec2
  ): Vec2 => snapMovedPoint(origin, delta, positionGrid(ctx, input.modifiers.alt));

  const applyDelta = (ctx: ToolContext, current: GestureState, input: PointerInput): void => {
    const delta = deltaFor(current, input);
    if (current.mode.kind === 'layer') {
      const primary = current.mode.targets.find((target) => target.primary)!;
      const primaryNext = nextLayerPosition(ctx, primary.origin, input, delta);
      const effectiveDelta = { x: primaryNext.x - primary.origin.x, y: primaryNext.y - primary.origin.y };
      for (const target of current.mode.targets) {
        ctx.setLayerTransformOverride(target.id, {
          x: target.origin.x + effectiveDelta.x,
          y: target.origin.y + effectiveDelta.y,
        });
      }
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
      // Enter commits the float; engine Escape abandons it regardless of active tool.
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
        if (state.mode.kind === 'lift') {
          state.mode = liftOnDrag(ctx, state.mode);
        }
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
      if (current.mode.kind === 'none' || current.mode.kind === 'lift') {
        // No movable layer is selected — nothing to commit.
        return;
      }
      if (current.mode.kind === 'float') {
        // Retain float movement for one eventual history entry across drags; floats never snap.
        applyDelta(ctx, current, input);
        return;
      }

      const primary = current.mode.targets.find((target) => target.primary)!;
      const primaryNext = nextLayerPosition(ctx, primary.origin, input, deltaFor(current, input));
      const effectiveDelta = { x: primaryNext.x - primary.origin.x, y: primaryNext.y - primary.origin.y };

      if (effectiveDelta.x === 0 && effectiveDelta.y === 0) {
        // Drop zero-result previews, including movement snapped back to origin, without committing.
        for (const target of current.mode.targets) {
          ctx.setLayerTransformOverride(target.id, null);
        }
        return;
      }

      const next = current.mode.targets.map((target) => ({
        id: target.id,
        x: target.origin.x + effectiveDelta.x,
        y: target.origin.y + effectiveDelta.y,
      }));
      if (current.mode.targets.length === 1) {
        ctx.commitStructural(
          'Move layer',
          { id: primary.id, patch: { transform: { x: primaryNext.x, y: primaryNext.y } }, type: 'updateCanvasLayer' },
          {
            id: primary.id,
            patch: { transform: { x: primary.origin.x, y: primary.origin.y } },
            type: 'updateCanvasLayer',
          }
        );
      } else {
        ctx.commitStructural(
          'Move layers',
          { type: 'setCanvasLayerPositions', updates: next },
          {
            type: 'setCanvasLayerPositions',
            updates: current.mode.targets.map((target) => ({ id: target.id, ...target.origin })),
          }
        );
      }
      // The committed transform now flows through the mirror; drop the preview.
      for (const target of current.mode.targets) {
        ctx.setLayerTransformOverride(target.id, null);
      }
    },
  };
};
