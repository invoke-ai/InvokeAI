/**
 * Shared brush/eraser gesture resolves or creates a target, drives a StrokeSession and commits/cancels. Only
 * gesture-start layer creation dispatches here; moves never dispatch. Fill/erase differ by blend and option
 * source. A Shift press on a layer with a remembered stroke end draws a straight segment from it; the end is kept
 * layer-local, so moving or transforming the layer carries it.
 */

import type { CanvasLayerEditRefusal } from '@workbench/canvas-engine/capabilities';
import type { CanvasLayerContract, CanvasRasterLayerContractV2 } from '@workbench/canvas-engine/contracts';
import type { SemanticLeaf } from '@workbench/canvas-engine/document-model/semanticLeaf';
import type { CanvasNodeInsertionAnchor } from '@workbench/canvas-engine/document/insertionAnchors';
import type { Mat2d, PointerInput } from '@workbench/canvas-engine/types';

import { lookupDocumentLeaf } from '@workbench/canvas-engine/document-model/documentModel';
import { getDocumentLayer, getDocumentLeaves } from '@workbench/canvas-engine/document/documentIndex';
import { isLayerTransparencyLocked, leafEditRefusal } from '@workbench/canvas-engine/document/layerEligibility';
import { isMaskLayer } from '@workbench/canvas-engine/document/sources';
import { applyToPoint, fromTRS, invert } from '@workbench/canvas-engine/math/mat2d';

import type { StrokeEdit, Tool, ToolContext } from './tool';

import { MASK_PAINT_COLOR } from './paintConstants';
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

/** The target layer's document matrix; null is identity (a new or materialized layer). */
const targetMatrix = ({ transform }: PaintTarget): Mat2d | null =>
  transform
    ? fromTRS({ x: transform.x, y: transform.y }, transform.rotation, transform.scaleX, transform.scaleY)
    : null;

/** Reports why the selected layer refused the press, naming it for the notice. */
const refuse = (
  ctx: ToolContext,
  layer: CanvasLayerContract,
  status: CanvasLayerEditRefusal['status'],
  extra: Pick<CanvasLayerEditRefusal, 'canPaintNewLayer'> = {}
): null => {
  ctx.reportEditRefusal?.({ layerId: layer.id, layerName: layer.name, status, ...extra });
  return null;
};

/**
 * True when the layer's durable pixels are not in the cache yet. Painting now would grow a transparent stroke-sized
 * cache over them, so the press asks for them and says the layer is loading; the next press can edit the complete
 * pixels. A durable source can legitimately have no cache while disabled or outside the current frame.
 */
const awaitsPixels = (ctx: ToolContext, layerId: string, hasDurablePixels: boolean): boolean => {
  if (!hasDurablePixels) {
    return false;
  }
  const entry = ctx.layers.get(layerId);
  if (entry && !entry.stale) {
    return false;
  }
  ctx.requestLayerRasterization?.(layerId);
  return true;
};

/**
 * Why a press of `tool` on the selected leaf would be refused before it starts, without side effects: drives the
 * not-allowed cursor. Loading pixels are not predicted; the press itself reports them.
 */
const predictRefusal = (
  leaf: SemanticLeaf | null,
  tool: PaintToolSpec['id']
): CanvasLayerEditRefusal['status'] | null => {
  const selected = leaf?.layer;
  if (!leaf || !selected) {
    return null;
  }
  const paintsInPlace =
    selected.type === 'raster'
      ? selected.source.type === 'paint' || selected.source.type === 'image'
      : selected.type === 'control' || isMaskLayer(selected);
  if (!paintsInPlace) {
    return null;
  }
  const refusal = leafEditRefusal(leaf);
  if (refusal) {
    return refusal;
  }
  return tool === 'eraser' && isLayerTransparencyLocked(selected) ? 'layer-alpha-locked' : null;
};

/** Resolves (or auto-creates) the paint target for a gesture, or `null` to no-op. */
const resolveTarget = (ctx: ToolContext, tool: PaintToolSpec['id']): PaintTarget | null => {
  const doc = ctx.getDocument();
  if (!doc) {
    return null;
  }
  const leaf = doc.selectedLayerId ? lookupDocumentLeaf(doc, doc.selectedLayerId) : null;
  const selected = leaf?.layer;

  if (leaf && selected && selected.type === 'raster' && selected.source.type === 'paint') {
    // Locked/hidden paint targets refuse rather than silently creating another layer.
    const refusal = leafEditRefusal(leaf);
    if (refusal) {
      return refuse(ctx, selected, refusal);
    }
    if (awaitsPixels(ctx, selected.id, selected.source.bitmap !== null)) {
      return refuse(ctx, selected, 'layer-loading');
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

  if (leaf && selected?.type === 'raster' && selected.source.type === 'image') {
    // The brush and eraser paint a raster image in place, materializing it as an undoable paint layer under its
    // transparency lock. A locked or hidden image refuses (the brush offers a new paint layer instead of making one
    // unasked), as does one whose pixels have not loaded; the eraser also refuses a transparency lock. A rasterized
    // image whose spatial adjustments have not landed refuses `not-ready` (or names the stack that cannot run).
    const refusal = leafEditRefusal(leaf);
    if (refusal) {
      return refuse(ctx, selected, refusal, tool === 'brush' ? { canPaintNewLayer: true } : {});
    }
    const transparencyLocked = isLayerTransparencyLocked(selected);
    if (tool === 'eraser' && transparencyLocked) {
      return refuse(ctx, selected, 'layer-alpha-locked');
    }
    if (awaitsPixels(ctx, selected.id, true)) {
      return refuse(ctx, selected, 'layer-loading');
    }
    const transaction = ctx.beginPixelEdit?.(selected.id) ?? null;
    // No transform on purpose: materialization BAKES the layer's placement
    // into document-space pixels and resets the layer to identity, so the
    // session's document coordinates already are the cache's coordinates.
    return transaction
      ? editTarget(transaction, transaction.layerId, tool === 'brush' ? { transparencyLocked } : {})
      : null;
  }

  if (leaf && selected && isMaskLayer(selected)) {
    // Paint mask alpha directly, adding/removing coverage. Locked/hidden masks refuse and never auto-create
    // paint layers.
    const refusal = leafEditRefusal(leaf);
    if (refusal) {
      return refuse(ctx, selected, refusal);
    }
    if (awaitsPixels(ctx, selected.id, selected.mask.bitmap !== null)) {
      return refuse(ctx, selected, 'layer-loading');
    }
    const edit = ctx.beginStrokeEdit();
    return edit
      ? editTarget(edit, selected.id, { color: MASK_PAINT_COLOR, forceOpaque: true, transform: selected.transform })
      : null;
  }

  if (leaf && selected?.type === 'control') {
    const refusal = leafEditRefusal(leaf);
    if (refusal) {
      return refuse(ctx, selected, refusal);
    }
    const durable = selected.source.type === 'image' || (selected.source.type === 'paint' && !!selected.source.bitmap);
    if (awaitsPixels(ctx, selected.id, durable)) {
      return refuse(ctx, selected, 'layer-loading');
    }
    const transaction = ctx.beginPixelEdit?.(selected.id) ?? null;
    // No transform on purpose: see the raster-image branch above.
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
    // Not-allowed over a selection the press would refuse; the ring still shows the brush size.
    cursor: (ctx) => {
      const doc = ctx.getDocument();
      const leaf = doc?.selectedLayerId ? lookupDocumentLeaf(doc, doc.selectedLayerId) : null;
      return !session && predictRefusal(leaf, spec.id) ? 'not-allowed' : 'crosshair';
    },
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
        const layer = getDocumentLayer(ctx.getDocument(), resolvedTarget.layerId);
        if (layer) {
          refuse(ctx, layer, 'layer-alpha-locked');
        }
        return;
      }
      const composite = resolvedTarget.transparencyLocked && spec.id === 'brush' ? 'source-atop' : spec.composite;
      const layerTransform = targetMatrix(resolvedTarget);
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
        // Shift continues from the layer's last stroke end, placed through the layer's current transform: both points
        // take the press pressure, so the segment renders as a line rather than collapsing to a tap.
        const localStart = input.modifiers.shift ? ctx.paintLineEnds?.get(target.layerId) : undefined;
        const lineStart = localStart && layerTransform ? applyToPoint(layerTransform, localStart) : localStart;
        session.addPoints(lineStart ? [{ ...input, documentPoint: lineStart }, input] : [input]);
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
        if (published) {
          // Kept layer-local, in the space the stroke painted; the press already refused a non-invertible matrix.
          const matrix = targetMatrix(activeTarget);
          const toLocal = matrix ? invert(matrix) : null;
          ctx.paintLineEnds?.set(
            activeTarget.layerId,
            toLocal ? applyToPoint(toLocal, input.documentPoint) : { ...input.documentPoint }
          );
        } else {
          activeTarget.cancel();
        }
      }
    },
  };
};
