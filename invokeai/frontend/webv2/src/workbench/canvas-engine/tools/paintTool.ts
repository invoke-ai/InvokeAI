/** Only gesture-start layer creation dispatches; moves never do. */

import type { CanvasLayerEditRefusal } from '@workbench/canvas-engine/capabilities';
import type { CanvasLayerContract, CanvasRasterLayerContractV2 } from '@workbench/canvas-engine/contracts';
import type { SemanticLeaf } from '@workbench/canvas-engine/document-model/semanticLeaf';
import type { CanvasNodeInsertionAnchor } from '@workbench/canvas-engine/document/insertionAnchors';
import type { Mat2d, PointerInput } from '@workbench/canvas-engine/types';

import { lookupDocumentLeaf } from '@workbench/canvas-engine/document-model/documentModel';
import { getDocumentLayer, getDocumentLeaves, getDocumentNode } from '@workbench/canvas-engine/document/documentIndex';
import { isLayerTransparencyLocked } from '@workbench/canvas-engine/document/layerEligibility';
import { isMaskLayer } from '@workbench/canvas-engine/document/sources';
import { editsPixelsInPlace, leafPaintRefusal } from '@workbench/canvas-engine/editing/controlPixelEdit';
import { applyToPoint, fromTRS, invert } from '@workbench/canvas-engine/math/mat2d';

import type { StrokeEdit, Tool, ToolContext } from './tool';

import { MASK_PAINT_COLOR } from './paintConstants';
import { createStrokeSession, type StrokeSession } from './strokeSession';

const PRIMARY_BUTTON = 1;

export interface PaintToolSpec {
  id: 'brush' | 'eraser';
  composite: 'source-over' | 'destination-out';
  size(ctx: ToolContext): number;
  opacity(ctx: ToolContext): number;
  /** `null` for the eraser. */
  color(ctx: ToolContext): string;
  thinning(ctx: ToolContext): number;
  /** Absent means 1 (crisp). */
  hardness?(ctx: ToolContext): number;
  /** Absent means never (eraser). */
  pressureOpacity?(ctx: ToolContext): boolean;
}

interface PaintTarget {
  layerId: string;
  edit: StrokeEdit;
  cancel(): void;
  createdLayer?: { layer: CanvasLayerContract; anchor: CanvasNodeInsertionAnchor };
  /** Mask RGB is irrelevant: compositing colorizes alpha. */
  color?: string;
  /** Transparency-locked raster paint layers only, never masks: the brush uses `source-atop`, the eraser is refused. */
  transparencyLocked?: boolean;
  /** Masks force opacity 1; partial alpha would silently attenuate generation coverage. */
  forceOpaque?: boolean;
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

/** Null is identity (a new or materialized layer). */
const targetMatrix = ({ transform }: PaintTarget): Mat2d | null =>
  transform
    ? fromTRS({ x: transform.x, y: transform.y }, transform.rotation, transform.scaleX, transform.scaleY)
    : null;

const refuse = (
  ctx: ToolContext,
  layer: Pick<CanvasLayerContract, 'id' | 'name'>,
  status: CanvasLayerEditRefusal['status'],
  extra: Pick<CanvasLayerEditRefusal, 'canPaintNewLayer'> = {}
): null => {
  ctx.reportEditRefusal?.({ layerId: layer.id, layerName: layer.name, status, ...extra });
  return null;
};

/** Painting now would grow a transparent cache over the durable pixels, so the press asks for them instead. */
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

/** Side-effect free; drives the not-allowed cursor. */
const predictsRefusal = (leaf: SemanticLeaf | null, tool: PaintToolSpec['id']): boolean => {
  const selected = leaf?.layer;
  if (!leaf || !selected || !editsPixelsInPlace(selected)) {
    // A control layer with no pixels refuses both tools; elsewhere the brush paints a new layer instead.
    return tool === 'eraser' || selected?.type === 'control';
  }
  return leafPaintRefusal(leaf) !== null || (tool === 'eraser' && isLayerTransparencyLocked(selected));
};

const resolveTarget = (ctx: ToolContext, tool: PaintToolSpec['id']): PaintTarget | null => {
  const doc = ctx.getDocument();
  if (!doc) {
    return null;
  }
  const leaf = doc.selectedLayerId ? lookupDocumentLeaf(doc, doc.selectedLayerId) : null;
  const selected = leaf?.layer;

  if (selected?.type === 'vector') {
    return null;
  }

  if (leaf && selected && selected.type === 'raster' && selected.source.type === 'paint') {
    // Locked/hidden paint targets refuse rather than silently creating another layer.
    const refusal = leafPaintRefusal(leaf);
    if (refusal) {
      return refuse(ctx, selected, refusal);
    }
    if (awaitsPixels(ctx, selected.id, selected.source.bitmap !== null)) {
      return refuse(ctx, selected, 'layer-loading');
    }
    const edit = ctx.beginStrokeEdit();
    return edit
      ? editTarget(edit, selected.id, {
          transform: selected.transform,
          transparencyLocked: selected.isTransparencyLocked === true,
        })
      : null;
  }

  if (leaf && selected?.type === 'raster' && selected.source.type === 'image') {
    // Raster images paint in place, materialized as an undoable paint layer under their transparency lock.
    const refusal = leafPaintRefusal(leaf);
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
    // No transform on purpose: materialization BAKES the layer's placement into document-space pixels and resets the
    // layer to identity, so the session's document coordinates already are the cache's coordinates.
    return transaction
      ? editTarget(transaction, transaction.layerId, tool === 'brush' ? { transparencyLocked } : {})
      : null;
  }

  if (leaf && selected && isMaskLayer(selected)) {
    const refusal = leafPaintRefusal(leaf);
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

  if (leaf && selected?.type === 'control' && editsPixelsInPlace(selected)) {
    const refusal = leafPaintRefusal(leaf);
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

  if (selected?.type === 'control') {
    return refuse(ctx, selected, 'layer-unpaintable');
  }
  if (tool === 'eraser') {
    const node = getDocumentNode(doc, doc.selectedLayerId);
    ctx.reportEditRefusal?.(
      node ? { layerId: node.id, layerName: node.name, status: 'erase-unsupported' } : { status: 'erase-no-target' }
    );
    return null;
  }

  // Admit the stroke before the layer exists, so a refusal leaves the document untouched.
  const edit = ctx.beginStrokeEdit();
  if (!edit) {
    return null;
  }
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

  const entry = ctx.layers.getOrCreateRect(layerId, { height: 0, width: 0, x: 0, y: 0 });
  entry.stale = false;
  return {
    // Gesture-start creation stays outside history until a stroke commits; roll it back otherwise.
    cancel: () => {
      edit.cancel();
      ctx.layers.delete(layerId);
      ctx.dispatch({ ids: [layerId], type: 'removeCanvasLayers' });
      // The reducer's nearest-neighbour fallback would otherwise select whatever sits at the top, not what the user had
      // selected when the gesture began.
      ctx.dispatch({ id: previousSelectedLayerId, type: 'setCanvasSelectedLayer' });
    },
    createdLayer: { anchor, layer },
    edit,
    layerId,
  };
};

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
    cursor: (ctx) => {
      const doc = ctx.getDocument();
      const leaf = doc?.selectedLayerId ? lookupDocumentLeaf(doc, doc.selectedLayerId) : null;
      return !session && (leaf?.layer.type === 'vector' || predictsRefusal(leaf, spec.id))
        ? 'not-allowed'
        : 'crosshair';
    },
    editsSelectedLayer: (ctx) => {
      const doc = ctx.getDocument();
      const leaf = doc?.selectedLayerId ? lookupDocumentLeaf(doc, doc.selectedLayerId) : null;
      return !!leaf && editsPixelsInPlace(leaf.layer) && !predictsRefusal(leaf, spec.id);
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
          clipMask: ctx.getSelectionMask?.() ?? null,
          clipRect: ctx.getStrokeClipRect?.() ?? null,
          color: target.color ?? spec.color(ctx),
          composite,
          // Mask strokes are an all-or-nothing stencil; feathering would attenuate the denoise strength.
          hardness: target.forceOpaque ? 1 : (spec.hardness?.(ctx) ?? 1),
          createdLayer: target.createdLayer ?? null,
          ctx,
          layerId: target.layerId,
          layerTransform,
          edit: target.edit,
          onRenderError: () => abortSession(),
          // The session restored the layer; the target still removes a layer it created.
          onRefused: () => abortSession(),
          opacity: target.forceOpaque ? 1 : spec.opacity(ctx),
          // Pressure may not thin a mask stroke either.
          pressureOpacity: !target.forceOpaque && (spec.pressureOpacity?.(ctx) ?? false),
          size: spec.size(ctx),
          thinning: spec.thinning(ctx),
          tool: spec.id,
        });
        // Shift continues from the layer's last stroke end; both points take the press pressure.
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
