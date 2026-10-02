/**
 * Gradient drag defines linear start/end or radial center/radius. Editable selected gradients update placement in
 * layer-local space while preserving kind/stops/extent; locked or hidden gradients refuse. Otherwise create a
 * bbox-sized gradient from options. Preview/cancel never dispatch; one real drag makes one commit. Selection masks
 * do not clip parametric gradients.
 */

import type {
  CanvasLayerBaseContract,
  CanvasLayerSourceContract,
  CanvasRasterLayerContractV2,
} from '@workbench/canvas-engine/contracts';
import type { CanvasProjectMutation } from '@workbench/canvas-engine/mutationContracts';
import type { Vec2 } from '@workbench/canvas-engine/types';

import { lookupDocumentLeaf } from '@workbench/canvas-engine/document-model/documentModel';
import { getDocumentLeaves } from '@workbench/canvas-engine/document/documentIndex';
import { isLeafEditable } from '@workbench/canvas-engine/document/layerEligibility';
import { applyToPoint, invert } from '@workbench/canvas-engine/math/mat2d';

import type { Tool, ToolContext } from './tool';

import { layerMatrix } from './moveHitTest';

/** Bit for the primary (usually left) mouse button in `PointerEvent.buttons`. */
const PRIMARY_BUTTON = 1;

/** Screen-space distance (CSS px) the pointer must travel before a press becomes a drag. */
export const GRADIENT_DRAG_THRESHOLD_PX = 3;

interface GestureState {
  /** What the drag will place: the selected gradient's kind, else the tool option. */
  kind: 'linear' | 'radial';
  startDoc: Vec2;
  startScreen: Vec2;
  moved: boolean;
}

/** Degrees of the vector from `start` to `end` (0° = left→right). */
const angleFromDrag = (start: Vec2, end: Vec2): number =>
  (Math.atan2(end.y - start.y, end.x - start.x) * 180) / Math.PI;

/**
 * The placement a drag from `start` to `end` (layer-local px) gives a gradient:
 * a linear ramp is centered on the drag's midpoint and as long as the drag; a
 * radial one is centered on the press point with the drag length as radius.
 */
export const placementFromDrag = (
  kind: 'linear' | 'radial',
  start: Vec2,
  end: Vec2
): { angle: number; center: Vec2; span: number } => ({
  angle: angleFromDrag(start, end),
  center: kind === 'radial' ? start : { x: (start.x + end.x) / 2, y: (start.y + end.y) / 2 },
  span: Math.max(1, Math.hypot(end.x - start.x, end.y - start.y)),
});

/** Maps a document-space drag into a layer's local space; null when the transform is degenerate. */
const toLayerLocal = (
  transform: CanvasLayerBaseContract['transform'],
  start: Vec2,
  end: Vec2
): [start: Vec2, end: Vec2] | null => {
  const inverse = invert(layerMatrix(transform));
  return inverse ? [applyToPoint(inverse, start), applyToPoint(inverse, end)] : null;
};

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
      const doc = ctx.getDocument();
      if (!doc) {
        return;
      }
      const selected = doc.selectedLayerId ? lookupDocumentLeaf(doc, doc.selectedLayerId)?.layer : null;
      const kind =
        selected?.type === 'raster' && selected.source.type === 'gradient'
          ? selected.source.kind
          : ctx.stores.gradientOptions.get().kind;
      state = { kind, moved: false, startDoc: input.documentPoint, startScreen: input.screenPoint };
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
      ctx.stores.gradientPreview.set({ end: input.documentPoint, kind: state.kind, start: state.startDoc });
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
      const leaf = doc.selectedLayerId ? lookupDocumentLeaf(doc, doc.selectedLayerId) : null;
      const selected = leaf?.layer;

      if (leaf && selected && selected.type === 'raster' && selected.source.type === 'gradient') {
        // Edit the selected gradient layer — unless it's locked/disabled (no-op).
        const local = isLeafEditable(leaf)
          ? toLayerLocal(selected.transform, current.startDoc, input.documentPoint)
          : null;
        if (!local) {
          clearPreview(ctx);
          return;
        }
        const old = selected.source;
        const forward: CanvasProjectMutation = {
          id: selected.id,
          source: { ...old, ...placementFromDrag(old.kind, ...local) },
          type: 'updateCanvasLayerSource',
        };
        const inverse: CanvasProjectMutation = { id: selected.id, source: old, type: 'updateCanvasLayerSource' };
        ctx.commitStructural('Edit gradient', forward, inverse);
        clearPreview(ctx);
        return;
      }

      // New gradients use bbox size and origin; subtract that origin to map drag coordinates locally.
      const options = ctx.stores.gradientOptions.get();
      // The built-in FG→BG preset resolves the pair now; custom stops are
      // explicit and independent of later pair edits.
      const pair = ctx.stores.colorPair.get();
      const stops =
        options.preset === 'pair'
          ? [
              { color: `${pair.foreground}ff`, offset: 0 },
              { color: `${pair.background}ff`, offset: 1 },
            ]
          : options.stops.map((stop) => ({ ...stop }));
      const layerId = ctx.createLayerId();
      const origin = { x: doc.bbox.x, y: doc.bbox.y };
      const source: CanvasLayerSourceContract = {
        ...placementFromDrag(
          options.kind,
          { x: current.startDoc.x - origin.x, y: current.startDoc.y - origin.y },
          { x: input.documentPoint.x - origin.x, y: input.documentPoint.y - origin.y }
        ),
        height: doc.bbox.height,
        kind: options.kind,
        stops,
        type: 'gradient',
        width: doc.bbox.width,
      };
      const layer: CanvasRasterLayerContractV2 = {
        blendMode: 'normal',
        id: layerId,
        isEnabled: true,
        isLocked: false,
        name: `Gradient ${getDocumentLeaves(doc).length + 1}`,
        opacity: 1,
        source,
        transform: { rotation: 0, scaleX: 1, scaleY: 1, x: doc.bbox.x, y: doc.bbox.y },
        type: 'raster',
      };
      const forward: CanvasProjectMutation = {
        anchor: ctx.captureInsertionAnchor('raster', doc.selectedLayerId),
        layer,
        type: 'addCanvasLayer',
      };
      const inverse: CanvasProjectMutation = { ids: [layerId], type: 'removeCanvasLayers' };
      ctx.commitStructural('Add gradient', forward, inverse);
      clearPreview(ctx);
    },
  };
};
