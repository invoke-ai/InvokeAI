/**
 * Shared brush/eraser gesture resolves or creates a target, drives a StrokeSession and commits/cancels. Only
 * gesture-start layer creation dispatches here; moves never dispatch. Fill/erase differ by blend and option
 * source.
 */

import type { CanvasLayerContract, CanvasRasterLayerContractV2 } from '@workbench/canvas-engine/contracts';
import type { CanvasNodeInsertionAnchor } from '@workbench/canvas-engine/document/insertionAnchors';
import type { PointerInput } from '@workbench/canvas-engine/types';

import { lookupDocumentLeaf } from '@workbench/canvas-engine/document-model/documentModel';
import { getDocumentLeaves } from '@workbench/canvas-engine/document/documentIndex';
import { isLeafPaintable, isLayerTransparencyLocked } from '@workbench/canvas-engine/document/layerEligibility';
import { isMaskLayer } from '@workbench/canvas-engine/document/sources';
import { fromTRS, invert } from '@workbench/canvas-engine/math/mat2d';

import type { StrokeEdit, Tool, ToolContext } from './tool';

import { createStrokeSession, type StrokeSession } from './strokeSession';

/** Bit for the primary (usually left) mouse button in `PointerEvent.buttons`. */
const PRIMARY_BUTTON = 1;

/** The resolved config for one gesture, derived from the tool's options store. */
export interface PaintToolSpec {
  id: 'brush' | 'eraser';
  composite: 'source-over' | 'destination-out';
  /** Reads the current size (document units) from the options store. */
  size(ctx: ToolContext): number;
  /** Reads the current per-stroke opacity from the options store. */
  opacity(ctx: ToolContext): number;
  /** The fill color; `null` for the eraser (shape only). */
  color(ctx: ToolContext): string;
  /** Freehand thinning for this gesture; 0 disables pressure sensitivity. */
  thinning(ctx: ToolContext): number;
  /** Edge hardness in [0, 1]; absent means 1 (crisp). */
  hardness?(ctx: ToolContext): number;
  /** Whether pen pressure modulates alpha along the stroke. Absent means never (eraser). */
  pressureOpacity?(ctx: ToolContext): boolean;
}

/** Colour brush strokes paint into a MASK cache: an opaque stencil (only alpha matters). */
const MASK_STROKE_COLOR = '#ffffff';

/** The resolved paint target for a gesture. `createdLayer` is set only when auto-created. */
interface PaintTarget {
  layerId: string;
  /** The admitted edit that records the finished stroke. */
  edit: StrokeEdit;
  /** Ends the gesture unrecorded, removing a layer it auto-created. */
  cancel(): void;
  /** When the gesture auto-created its layer, the created contract + its anchor (for history). */
  createdLayer?: { layer: CanvasLayerContract; anchor: CanvasNodeInsertionAnchor };
  /** Optional gesture color override; mask RGB is irrelevant because compositing colorizes alpha. */
  color?: string;
  /**
   * True when the target is a transparency-LOCKED raster paint layer. The brush
   * then composites `source-atop` (colour only on existing pixels); the eraser is
   * refused (erasing would change the locked alpha). Never set for mask targets.
   */
  transparencyLocked?: boolean;
  /**
   * Masks force opacity one; partial brush alpha would silently attenuate generation coverage despite the tinted
   * preview.
   */
  forceOpaque?: boolean;
  /** Invert the committed layer transform to paint under the cursor; absence is identity for new layers. */
  transform?: CanvasLayerContract['transform'];
}

const editTarget = (
  edit: StrokeEdit,
  layerId: string,
  extra: Omit<PaintTarget, 'cancel' | 'edit' | 'layerId'> = {}
) => ({
  cancel: () => edit.cancel(),
  edit,
  layerId,
  ...extra,
});

/** Resolves (or auto-creates) the paint target for a gesture, or `null` to no-op. */
const resolveTarget = (ctx: ToolContext, tool: PaintToolSpec['id']): PaintTarget | null => {
  const doc = ctx.getDocument();
  if (!doc) {
    return null;
  }
  const leaf = doc.selectedLayerId ? lookupDocumentLeaf(doc, doc.selectedLayerId) : null;
  const selected = leaf?.layer;

  if (leaf && selected && selected.type === 'raster' && selected.source.type === 'paint') {
    // Locked/disabled paint targets refuse rather than silently creating another layer.
    if (!isLeafPaintable(leaf)) {
      return null;
    }
    if (selected.source.bitmap) {
      const entry = ctx.layers.get(selected.id);
      if (!entry || entry.stale) {
        // A durable paint source can legitimately have no cache while disabled
        // or outside the current frame. Never grow a transparent stroke-sized
        // cache over it; the compositor's rasterization pass will publish the
        // source first, and the next gesture can edit the complete pixels.
        ctx.requestLayerRasterization?.(selected.id);
        return null;
      }
    }
    const edit = ctx.beginStrokeEdit();
    // The cache (if any) keeps its current content extent; the stroke grows it.
    return edit
      ? editTarget(edit, selected.id, {
          transform: selected.transform,
          transparencyLocked: selected.isTransparencyLocked === true,
        })
      : null;
  }

  if (leaf && selected?.type === 'raster' && selected.source.type === 'image' && tool === 'eraser') {
    // Erase materializes image pixels as an undoable paint layer in place; locked, disabled or unready images
    // refuse without spawning.
    if (!isLeafPaintable(leaf) || isLayerTransparencyLocked(selected)) {
      return null;
    }
    const transaction = ctx.beginPixelEdit?.(selected.id) ?? null;
    // No transform on purpose: materialization BAKES the layer's placement
    // into document-space pixels and resets the layer to identity, so the
    // session's document coordinates already are the cache's coordinates.
    return transaction ? editTarget(transaction, transaction.layerId) : null;
  }

  if (leaf && selected && isMaskLayer(selected)) {
    // Paint mask alpha directly, adding/removing coverage. Locked/disabled masks refuse and never auto-create
    // paint layers.
    if (!isLeafPaintable(leaf)) {
      return null;
    }
    if (selected.mask.bitmap) {
      const entry = ctx.layers.get(selected.id);
      if (!entry || entry.stale) {
        ctx.requestLayerRasterization?.(selected.id);
        return null;
      }
    }
    const edit = ctx.beginStrokeEdit();
    return edit
      ? editTarget(edit, selected.id, { color: MASK_STROKE_COLOR, forceOpaque: true, transform: selected.transform })
      : null;
  }

  if (selected?.type === 'control') {
    const transaction = ctx.beginPixelEdit?.(selected.id) ?? null;
    // No transform on purpose: see the materialized-eraser branch above.
    return transaction ? editTarget(transaction, transaction.layerId) : null;
  }

  // Admit the stroke before the layer exists, so a refusal leaves the document untouched.
  const edit = ctx.beginStrokeEdit();
  if (!edit) {
    return null;
  }
  // Other eligible selections create a selected paint layer at the top via the sole gesture-start dispatch.
  const layerId = ctx.createLayerId();
  const previousSelectedLayerId = doc.selectedLayerId;
  const layer: CanvasRasterLayerContractV2 = {
    blendMode: 'normal',
    id: layerId,
    isEnabled: true,
    isLocked: false,
    name: `Layer ${getDocumentLeaves(doc).length + 1}`,
    opacity: 1,
    source: { bitmap: null, type: 'paint' },
    transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
    type: 'raster',
  };
  const anchor = ctx.captureInsertionAnchor('raster', doc.selectedLayerId);
  ctx.dispatch({ anchor, layer, type: 'addCanvasLayer' });

  // Mark the new zero-rect cache fresh to fence async rasterization; the stroke grows its bounds.
  const entry = ctx.layers.getOrCreateRect(layerId, { height: 0, width: 0, x: 0, y: 0 });
  entry.stale = false;
  return {
    // Gesture-start creation is outside history until a stroke commits. Roll it back for fully clipped strokes,
    // refusals, cancellation or mid-drag switching.
    cancel: () => {
      edit.cancel();
      ctx.layers.delete(layerId);
      ctx.dispatch({ ids: [layerId], type: 'removeCanvasLayers' });
      // The reducer's nearest-neighbour fallback would otherwise select whatever sits
      // at the top, not what the user had selected when the gesture began.
      ctx.dispatch({ id: previousSelectedLayerId, type: 'setCanvasSelectedLayer' });
    },
    createdLayer: { anchor, layer },
    edit,
    layerId,
  };
};

/** Creates a brush-family tool from its per-gesture {@link PaintToolSpec}. */
export const createPaintTool = (spec: PaintToolSpec): Tool => {
  let session: StrokeSession | null = null;
  let target: PaintTarget | null = null;

  const cursorRadiusDoc = (ctx: ToolContext): number => spec.size(ctx) / 2;

  const updateCursorRing = (ctx: ToolContext, input: PointerInput): void => {
    ctx.setOverlayCursor({ point: input.documentPoint, radiusDoc: cursorRadiusDoc(ctx) });
    ctx.invalidate({ overlay: true });
  };

  const endSession = (): void => {
    session = null;
    target = null;
  };

  const abortSession = (): void => {
    const activeSession = session;
    const activeTarget = target;
    try {
      activeSession?.cancel();
    } finally {
      try {
        activeTarget?.cancel();
      } finally {
        endSession();
      }
    }
  };

  return {
    cursor: () => 'crosshair',
    id: spec.id,
    onDeactivate: (ctx, opts) => {
      try {
        if ((session || target) && !opts?.temporary) {
          abortSession();
        }
      } finally {
        ctx.setOverlayCursor(null);
        ctx.invalidate({ overlay: true });
      }
    },
    onPointerCancel: () => {
      if (session || target) {
        abortSession();
      }
    },
    onPointerDown: (ctx, input) => {
      if (session || (input.buttons & PRIMARY_BUTTON) === 0) {
        return;
      }
      const resolvedTarget = resolveTarget(ctx, spec.id);
      updateCursorRing(ctx, input);
      if (!resolvedTarget) {
        return;
      }
      // Every refusal below ends the admitted edit (and removes a layer the gesture created).
      if (resolvedTarget.transparencyLocked && spec.id === 'eraser') {
        resolvedTarget.cancel();
        return;
      }
      const composite = resolvedTarget.transparencyLocked && spec.id === 'brush' ? 'source-atop' : spec.composite;
      const layerTransform = resolvedTarget.transform
        ? fromTRS(
            { x: resolvedTarget.transform.x, y: resolvedTarget.transform.y },
            resolvedTarget.transform.rotation,
            resolvedTarget.transform.scaleX,
            resolvedTarget.transform.scaleY
          )
        : null;
      if (layerTransform && !invert(layerTransform)) {
        // A zero-scale layer has no paintable geometry to invert into.
        resolvedTarget.cancel();
        return;
      }
      target = resolvedTarget;
      try {
        session = createStrokeSession({
          // Capture selection clipping once per gesture; no selection avoids per-point mask work.
          clipMask: ctx.getSelectionMask?.() ?? null,
          clipRect: ctx.getStrokeClipRect?.() ?? null,
          color: target.color ?? spec.color(ctx),
          composite,
          // Mask strokes are an all-or-nothing stencil; a feathered edge would
          // silently attenuate the denoise strength.
          hardness: target.forceOpaque ? 1 : (spec.hardness?.(ctx) ?? 1),
          createdLayer: target.createdLayer ?? null,
          ctx,
          layerId: target.layerId,
          layerTransform,
          edit: target.edit,
          // A deferred frame render fails outside this handler; abandon the stroke there too.
          onRenderError: () => abortSession(),
          // The session restored the layer; the target still removes a layer it created.
          onRefused: () => abortSession(),
          opacity: target.forceOpaque ? 1 : spec.opacity(ctx),
          // A mask stroke is an all-or-nothing alpha stencil, so pressure must not thin it —
          // a partially-transparent mask would silently attenuate the denoise strength.
          pressureOpacity: !target.forceOpaque && (spec.pressureOpacity?.(ctx) ?? false),
          size: spec.size(ctx),
          thinning: spec.thinning(ctx),
          tool: spec.id,
        });
        session.addPoints([input]);
      } catch {
        abortSession();
      }
    },
    onPointerMove: (ctx, input, batch) => {
      updateCursorRing(ctx, input);
      if (session) {
        try {
          session.addPoints(batch);
        } catch (error) {
          abortSession();
          throw error;
        }
      }
    },
    onPointerUp: (ctx, input) => {
      updateCursorRing(ctx, input);
      if (session && target) {
        const activeSession = session;
        const activeTarget = target;
        let published: boolean;
        try {
          published = activeSession.commit((event) => activeTarget.edit.commit(event));
        } catch (error) {
          abortSession();
          throw error;
        }
        endSession();
        if (!published) {
          activeTarget.cancel();
        }
      }
    },
  };
};
