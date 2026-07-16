/**
 * The gradient tool: drag to set a gradient's ANGLE from the drag vector.
 *
 * Interaction contract (CANVAS_PLAN Phase 6.1):
 * - **Pointer-down** (primary button) starts a gesture at the press point.
 * - **Pointer-move** (past a small threshold) updates a transient overlay
 *   preview (`stores.gradientPreview`, the drag vector) — it never dispatches.
 * - **Commit** (pointer-up after a real drag): exactly ONE commit.
 *   - A gradient layer is selected (unlocked + visible) → one `commitStructural`
 *     with `updateCanvasLayerSource` (new/old angle; kind + stops preserved).
 *   - Otherwise → create a new bbox-sized gradient layer (angle from the drag,
 *     kind + stops from the tool options, extent = the generation frame) via
 *     `addCanvasLayer`.
 *   - A selected gradient layer that is locked/hidden is a no-op (don't silently
 *     spawn a new layer over it), mirroring the paint tool's locked-target rule.
 * - **Cancel** (Esc / pointercancel): drops the preview, no dispatch.
 *
 * Gradient layers are content-sized via the contract's explicit `width`/`height`
 * extent — set at creation (bbox-sized) and preserved across angle edits — so
 * only the ANGLE is derived from the drag, not a bounding box. The selection
 * mask does NOT constrain gradient layers (parametric, not pixels).
 *
 * Zero React, zero import-time side effects.
 */

import type { Vec2 } from '@workbench/canvas-engine/types';
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';
import type { CanvasLayerSourceContract, CanvasRasterLayerContractV2 } from '@workbench/types';

import type { Tool, ToolContext } from './tool';

/** Bit for the primary (usually left) mouse button in `PointerEvent.buttons`. */
const PRIMARY_BUTTON = 1;

/** Screen-space distance (CSS px) the pointer must travel before a press becomes a drag. */
export const GRADIENT_DRAG_THRESHOLD_PX = 3;

interface GestureState {
  startDoc: Vec2;
  startScreen: Vec2;
  moved: boolean;
}

/** Degrees of the vector from `start` to `end` (0° = left→right). */
export const angleFromDrag = (start: Vec2, end: Vec2): number =>
  (Math.atan2(end.y - start.y, end.x - start.x) * 180) / Math.PI;

/** Creates a fresh gradient tool with its own gesture state. */
export const createGradientTool = (): Tool => {
  let state: GestureState | null = null;

  const clearPreview = (ctx: ToolContext): void => {
    ctx.stores.gradientPreview.set(null);
    ctx.invalidate({ overlay: true });
  };

  return {
    cursor: () => 'crosshair',
    id: 'gradient',
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
        if (Math.hypot(dxs, dys) < GRADIENT_DRAG_THRESHOLD_PX) {
          return;
        }
        state.moved = true;
      }
      ctx.stores.gradientPreview.set({ end: input.documentPoint, start: state.startDoc });
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

      const doc = ctx.getDocument();
      if (!doc) {
        clearPreview(ctx);
        return;
      }
      const angle = angleFromDrag(current.startDoc, input.documentPoint);
      const selected = doc.selectedLayerId ? doc.layers.find((layer) => layer.id === doc.selectedLayerId) : undefined;

      if (selected && selected.type === 'raster' && selected.source.type === 'gradient') {
        // Edit the selected gradient layer — unless it's locked/hidden (no-op).
        if (selected.isLocked || !selected.isEnabled) {
          clearPreview(ctx);
          return;
        }
        const old = selected.source;
        // A radial gradient's rendering ignores `angle` entirely — dragging on
        // one would only ever change that inert field, producing a commit with
        // zero visual effect and a useless history entry. Skip it.
        if (old.kind === 'radial') {
          clearPreview(ctx);
          return;
        }
        const forward: CanvasProjectMutation = {
          id: selected.id,
          source: { ...old, angle },
          type: 'updateCanvasLayerSource',
        };
        const inverse: CanvasProjectMutation = { id: selected.id, source: old, type: 'updateCanvasLayerSource' };
        ctx.commitStructural('Edit gradient', forward, inverse);
        clearPreview(ctx);
        return;
      }

      // Create a new bbox-sized gradient layer. The document rect is retired, so
      // creation covers the generation frame (bbox): the extent is the bbox size,
      // positioned at the bbox origin via the layer transform. Angle-drag edits on
      // an existing gradient preserve its extent (the `...old` spread above).
      const options = ctx.stores.gradientOptions.get();
      const layerId = ctx.createLayerId();
      const source: CanvasLayerSourceContract = {
        angle,
        height: doc.bbox.height,
        kind: options.kind,
        stops: options.stops.map((stop) => ({ ...stop })),
        type: 'gradient',
        width: doc.bbox.width,
      };
      const layer: CanvasRasterLayerContractV2 = {
        blendMode: 'normal',
        id: layerId,
        isEnabled: true,
        isLocked: false,
        name: `Gradient ${doc.layers.length + 1}`,
        opacity: 1,
        source,
        transform: { rotation: 0, scaleX: 1, scaleY: 1, x: doc.bbox.x, y: doc.bbox.y },
        type: 'raster',
      };
      const forward: CanvasProjectMutation = { index: 0, layer, type: 'addCanvasLayer' };
      const inverse: CanvasProjectMutation = { ids: [layerId], type: 'removeCanvasLayers' };
      ctx.commitStructural('Add gradient', forward, inverse);
      clearPreview(ctx);
    },
  };
};
