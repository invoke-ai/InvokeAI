/**
 * Shapes use box drags, polygon clicks or closed freehand. Commit pixels to an eligible selected paint layer or
 * mask (as opaque mask coverage) with selection/bbox clipping, otherwise create a parametric layer.
 * Locked/disabled/unready paint or masks refuse. Shift constrains box aspect; previews/cancellation do not
 * dispatch, and nonzero shapes produce one structural or stroke commit.
 */

import type { CanvasLayerSourceContract, CanvasRasterLayerContractV2 } from '@workbench/canvas-engine/contracts';
import type { SemanticLeaf } from '@workbench/canvas-engine/document-model/semanticLeaf';
import type { ShapeToolOptions } from '@workbench/canvas-engine/engineStores';
import type { CanvasProjectMutation } from '@workbench/canvas-engine/mutationContracts';
import type { PlacedSurface, Rect, Vec2 } from '@workbench/canvas-engine/types';

import { lookupDocumentLeaf } from '@workbench/canvas-engine/document-model/documentModel';
import { getDocumentLeaves } from '@workbench/canvas-engine/document/documentIndex';
import { isLayerTransparencyLocked, isLeafPaintable } from '@workbench/canvas-engine/document/layerEligibility';
import { isMaskLayer, maskAsPaintSource } from '@workbench/canvas-engine/document/sources';
import { polygonBounds } from '@workbench/canvas-engine/freehand';
import { identity, invert, multiply, translate } from '@workbench/canvas-engine/math/mat2d';
import { intersect, roundOut, transformBounds } from '@workbench/canvas-engine/math/rect';
import { drawShapeSource } from '@workbench/canvas-engine/render/rasterizers/shapeRasterizer';

import type { Tool, ToolContext } from './tool';

import { layerMatrix } from './moveHitTest';
import { MASK_PAINT_COLOR } from './paintConstants';
import {
  extendFreehandTrace,
  finishFreehandTrace,
  MIN_POLYLINE_POINTS,
  movePolyline,
  polylinePreview,
  pressPolyline,
  startFreehandTrace,
  startPolyline,
  type FreehandTrace,
  type PolylineSession,
} from './polylineSession';

type ShapeSource = Extract<CanvasLayerSourceContract, { type: 'shape' }>;

/** Bit for the primary (usually left) mouse button in `PointerEvent.buttons`. */
const PRIMARY_BUTTON = 1;

/** Screen-space distance (CSS px) the pointer must travel before a press becomes a drag. */
export const SHAPE_DRAG_THRESHOLD_PX = 3;

type Session =
  | { kind: 'drag'; startDoc: Vec2; startScreen: Vec2; moved: boolean }
  | { kind: 'polyline'; session: PolylineSession }
  | ({ kind: 'freehand' } & FreehandTrace);

const distance = (a: Vec2, b: Vec2): number => Math.hypot(a.x - b.x, a.y - b.y);

/** The integer, normalized document rect for a drag from `start` to `end`, optionally square-constrained. */
export const rectFromDrag = (start: Vec2, end: Vec2, square: boolean): Rect => {
  let dx = end.x - start.x;
  let dy = end.y - start.y;
  if (square) {
    const side = Math.max(Math.abs(dx), Math.abs(dy));
    dx = (dx < 0 ? -1 : 1) * side;
    dy = (dy < 0 ? -1 : 1) * side;
  }
  const x = Math.round(Math.min(start.x, start.x + dx));
  const y = Math.round(Math.min(start.y, start.y + dy));
  return { height: Math.round(Math.abs(dy)), width: Math.round(Math.abs(dx)), x, y };
};

/** A shape to place: its source plus the document rect it covers. */
interface PlacedShape {
  source: ShapeSource;
  rect: Rect;
}

/**
 * A polygon source for document-space vertices: the extent is the vertex
 * bounds, and the stored points are relative to that box's origin.
 */
export const polygonShapeFrom = (
  vertices: readonly Vec2[],
  style: Pick<ShapeSource, 'fill' | 'stroke' | 'strokeWidth'>
): PlacedShape | null => {
  const distinct = vertices.filter((point, index) => index === 0 || distance(point, vertices[index - 1]!) >= 1);
  const bounds = polygonBounds(distinct);
  if (distinct.length < MIN_POLYLINE_POINTS || bounds.width < 1 || bounds.height < 1) {
    return null;
  }
  const rect = roundOut(bounds);
  return {
    rect,
    source: {
      ...style,
      height: rect.height,
      kind: 'polygon',
      points: distinct.map((point) => ({ x: point.x - rect.x, y: point.y - rect.y })),
      type: 'shape',
      width: rect.width,
    },
  };
};

/** Where a finished shape went: pixels, nothing to draw, refused by the paint layer or mask, or neither at all. */
type PixelPlacement = 'placed' | 'nothing' | 'refused' | 'unsupported';

/** With fill off and no stroke to draw, a shape changes no pixel, so it records no step and uploads nothing. */
const drawsNothing = (source: ShapeSource): boolean =>
  source.fill === null && (source.stroke === null || source.strokeWidth <= 0);

/** A mask takes the shape's footprint as coverage: each enabled part paints opaque, whatever its color. */
const maskCoverageOf = (source: ShapeSource): ShapeSource => ({
  ...source,
  fill: source.fill === null ? null : MASK_PAINT_COLOR,
  stroke: source.stroke === null ? null : MASK_PAINT_COLOR,
});

/** Whether a session was started for the current kind option; a kind change mid-session drops it. */
const sessionFitsKind = (session: Session, kind: ShapeToolOptions['kind']): boolean =>
  kind === 'polygon'
    ? session.kind === 'polyline'
    : kind === 'freehand'
      ? session.kind === 'freehand'
      : session.kind === 'drag';

/** Creates a fresh shape tool with its own gesture state. */
export const createShapeTool = (): Tool => {
  let session: Session | null = null;

  const clearPreview = (ctx: ToolContext): void => {
    ctx.stores.shapePreview.set(null);
    ctx.stores.lassoPreview.set(null);
    ctx.invalidate({ overlay: true });
  };

  const reset = (ctx: ToolContext): void => {
    session = null;
    clearPreview(ctx);
  };

  /** The fill/stroke the next shape gets, resolved from the active pair now. */
  const styleFromOptions = (ctx: ToolContext): Pick<ShapeSource, 'fill' | 'stroke' | 'strokeWidth'> => {
    const options = ctx.stores.shapeOptions.get();
    const pair = ctx.stores.colorPair.get();
    return {
      fill: options.fillEnabled ? pair.foreground : null,
      stroke: options.strokeEnabled ? pair.background : null,
      strokeWidth: options.strokeWidth,
    };
  };

  /**
   * One pixel-shape stroke grows local bounds within selection/bbox clips, draws through the layer inverse and
   * applies the same masks as brush painting, on paint pixels or mask alpha alike.
   */
  const commitPixels = (ctx: ToolContext, leaf: SemanticLeaf, placed: PlacedShape): PixelPlacement => {
    const layer = leaf.layer;
    const paint = layer.type === 'raster' && layer.source.type === 'paint' ? layer.source : maskAsPaintSource(layer);
    if (!paint) {
      return 'unsupported';
    }
    if (drawsNothing(placed.source)) {
      return 'nothing';
    }
    if (!isLeafPaintable(leaf)) {
      return 'refused';
    }
    if (paint.bitmap) {
      const existing = ctx.layers.get(layer.id);
      if (!existing || existing.stale) {
        // The durable pixels are not in the cache yet: drawing now would lose them.
        ctx.requestLayerRasterization?.(layer.id);
        return 'refused';
      }
    }
    const toLocal = invert(layerMatrix(layer.transform));
    if (!toLocal) {
      return 'refused';
    }
    const clipMask: PlacedSurface | null = ctx.getSelectionMask?.() ?? null;
    const clipRect: Rect | null = ctx.getStrokeClipRect?.() ?? null;
    let docRect: Rect | null = placed.rect;
    if (clipMask) {
      docRect = intersect(docRect, clipMask.rect);
    }
    if (docRect && clipRect) {
      docRect = intersect(docRect, clipRect);
    }
    const dirtyRect = docRect ? roundOut(transformBounds(toLocal, docRect)) : null;
    if (!dirtyRect || dirtyRect.width < 1 || dirtyRect.height < 1) {
      return 'refused';
    }
    // Admitted before the first pixel changes: the before/after pair is the step's footprint.
    const edit = ctx.beginStrokeEdit(dirtyRect.width * dirtyRect.height * 8);
    if (!edit) {
      return 'refused';
    }
    const original = ctx.layers.peek(layer.id);
    const originalRect = original ? { ...original.rect } : null;
    let beforeImageData: ImageData | null = null;
    let sx = 0;
    let sy = 0;
    /** Puts the pre-shape pixels and cache extent back and ends the edit unrecorded. */
    const restore = (): void => {
      const entry = ctx.layers.peek(layer.id);
      if (entry && beforeImageData) {
        entry.surface.ctx.putImageData(beforeImageData, sx, sy);
      }
      if (originalRect) {
        ctx.layers.shrinkToRect(layer.id, originalRect);
      } else {
        ctx.layers.delete(layer.id);
      }
      ctx.notifyLayerPainted(layer.id);
      edit.cancel();
    };
    try {
      const entry = ctx.layers.growToRect(layer.id, dirtyRect);
      const surfaceCtx = entry.surface.ctx;
      sx = dirtyRect.x - entry.rect.x;
      sy = dirtyRect.y - entry.rect.y;
      beforeImageData = surfaceCtx.getImageData(sx, sy, dirtyRect.width, dirtyRect.height);

      // Draw into dirty-rect-local scratch so clipping and transparency lock apply in one composite; map through
      // layer inverse then scratch offset.
      const scratch = ctx.backend.createSurface(dirtyRect.width, dirtyRect.height);
      const draw = scratch.ctx;
      const { rect } = placed;
      const source = isMaskLayer(layer) ? maskCoverageOf(placed.source) : placed.source;
      const toScratch = multiply(translate(identity(), { x: -dirtyRect.x, y: -dirtyRect.y }), toLocal);
      draw.setTransform(toScratch.a, toScratch.b, toScratch.c, toScratch.d, toScratch.e, toScratch.f);
      drawShapeSource(draw, source, rect.x, rect.y, rect.width, rect.height);
      draw.globalCompositeOperation = 'destination-in';
      if (clipMask) {
        // The mask sits in document space, so it goes through the same mapping.
        draw.drawImage(clipMask.surface.canvas, clipMask.rect.x, clipMask.rect.y);
      }
      if (clipRect && (toLocal.b !== 0 || toLocal.c !== 0)) {
        // The dirty-rect clamp only bounds the AABB on a rotated/sheared layer;
        // keep exactly the pixels inside the document-space rect.
        draw.fillStyle = '#000';
        draw.beginPath();
        draw.rect(clipRect.x, clipRect.y, clipRect.width, clipRect.height);
        draw.fill();
      }
      surfaceCtx.save();
      surfaceCtx.setTransform(1, 0, 0, 1, 0, 0);
      surfaceCtx.globalCompositeOperation = isLayerTransparencyLocked(layer) ? 'source-atop' : 'source-over';
      surfaceCtx.drawImage(scratch.canvas, sx, sy);
      surfaceCtx.restore();
      const afterImageData = surfaceCtx.getImageData(sx, sy, dirtyRect.width, dirtyRect.height);
      if (!edit.commit({ afterImageData, beforeImageData, dirtyRect, layerId: layer.id, tool: 'shape' })) {
        restore();
        return 'refused';
      }
    } catch (error) {
      restore();
      throw error;
    }
    return 'placed';
  };

  const commitLayer = (ctx: ToolContext, placed: PlacedShape): void => {
    const doc = ctx.getDocument();
    if (!doc) {
      return;
    }
    const layerId = ctx.createLayerId();
    const layer: CanvasRasterLayerContractV2 = {
      blendMode: 'normal',
      id: layerId,
      isEnabled: true,
      isLocked: false,
      name: `Shape ${getDocumentLeaves(doc).length + 1}`,
      opacity: 1,
      source: placed.source,
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: placed.rect.x, y: placed.rect.y },
      type: 'raster',
    };
    const forward: CanvasProjectMutation = {
      anchor: ctx.captureInsertionAnchor('raster', doc.selectedLayerId),
      layer,
      type: 'addCanvasLayer',
    };
    const inverse: CanvasProjectMutation = { ids: [layerId], type: 'removeCanvasLayers' };
    ctx.commitStructural('Add shape', forward, inverse);
  };

  /**
   * Places a finished shape on the selected paint layer or mask when asked, else on a new layer. A refused
   * selected target places nothing rather than adding a layer the user did not ask for.
   */
  const commit = (ctx: ToolContext, placed: PlacedShape | null): void => {
    reset(ctx);
    const doc = ctx.getDocument();
    if (!placed || !doc) {
      return;
    }
    if (ctx.stores.shapeOptions.get().target === 'selected') {
      const leaf = doc.selectedLayerId ? lookupDocumentLeaf(doc, doc.selectedLayerId) : null;
      if (leaf && commitPixels(ctx, leaf, placed) !== 'unsupported') {
        return;
      }
    }
    commitLayer(ctx, placed);
  };

  const boxShape = (ctx: ToolContext, rect: Rect): PlacedShape | null => {
    const kind = ctx.stores.shapeOptions.get().kind;
    if (rect.width < 1 || rect.height < 1 || kind === 'polygon' || kind === 'freehand') {
      return null;
    }
    return { rect, source: { ...styleFromOptions(ctx), height: rect.height, kind, type: 'shape', width: rect.width } };
  };

  const closePolyline = (ctx: ToolContext, polyline: PolylineSession): void => {
    commit(ctx, polygonShapeFrom(polyline.points, styleFromOptions(ctx)));
  };

  return {
    cursor: () => (session?.kind === 'polyline' && session.session.closeArmed ? 'pointer' : 'crosshair'),
    id: 'shape',
    onDeactivate: (ctx, opts) => {
      // A modifier-hold switch (space → view to pan mid-polygon) keeps the session.
      if (!opts?.temporary) {
        reset(ctx);
      }
    },
    onKeyCommand: (ctx, command) => {
      if (!session) {
        return;
      }
      if (command === 'cancel') {
        reset(ctx);
      } else if (session.kind === 'polyline') {
        closePolyline(ctx, session.session);
      }
    },
    onPointerCancel: (ctx) => reset(ctx),
    onPointerDown: (ctx, input) => {
      if ((input.buttons & PRIMARY_BUTTON) === 0 || !ctx.getDocument()) {
        return;
      }
      const kind = ctx.stores.shapeOptions.get().kind;
      if (session && !sessionFitsKind(session, kind)) {
        reset(ctx);
      }
      if (kind === 'polygon') {
        if (session?.kind !== 'polyline') {
          session = { kind: 'polyline', session: startPolyline(input) };
        } else if (pressPolyline(ctx, session.session, input) === 'close') {
          closePolyline(ctx, session.session);
          return;
        }
        ctx.stores.lassoPreview.set(polylinePreview(session.session));
        ctx.invalidate({ overlay: true });
        return;
      }
      if (session) {
        return;
      }
      session =
        kind === 'freehand'
          ? { ...startFreehandTrace(input), kind: 'freehand' }
          : { kind: 'drag', moved: false, startDoc: input.documentPoint, startScreen: input.screenPoint };
    },
    onPointerMove: (ctx, input, batch) => {
      if (!session) {
        return;
      }
      const kind = ctx.stores.shapeOptions.get().kind;
      if (!sessionFitsKind(session, kind)) {
        reset(ctx);
        return;
      }
      if (session.kind === 'polyline') {
        if (movePolyline(ctx, session.session, input)) {
          ctx.updateCursor();
        }
        ctx.stores.lassoPreview.set(polylinePreview(session.session));
        ctx.invalidate({ overlay: true });
        return;
      }
      if (session.kind === 'freehand') {
        extendFreehandTrace(session, batch);
        ctx.stores.lassoPreview.set({ kind: 'freehand', points: session.points.slice() });
        ctx.invalidate({ overlay: true });
        return;
      }
      if (!session.moved) {
        const dxs = input.screenPoint.x - session.startScreen.x;
        const dys = input.screenPoint.y - session.startScreen.y;
        if (Math.hypot(dxs, dys) < SHAPE_DRAG_THRESHOLD_PX) {
          return;
        }
        session.moved = true;
      }
      if (kind !== 'polygon' && kind !== 'freehand') {
        const rect = rectFromDrag(session.startDoc, input.documentPoint, input.modifiers.shift);
        ctx.stores.shapePreview.set({ kind, rect });
        ctx.invalidate({ overlay: true });
      }
    },
    onPointerUp: (ctx, input) => {
      if (!session || session.kind === 'polyline') {
        return;
      }
      if (session.kind === 'freehand') {
        commit(ctx, polygonShapeFrom(finishFreehandTrace(session, input), styleFromOptions(ctx)));
        return;
      }
      if (!session.moved) {
        reset(ctx);
        return;
      }
      commit(ctx, boxShape(ctx, rectFromDrag(session.startDoc, input.documentPoint, input.modifiers.shift)));
    },
  };
};
