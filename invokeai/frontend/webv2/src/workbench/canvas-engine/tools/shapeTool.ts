/**
 * Shapes use box drags, polygon clicks or closed freehand. Place on = selected layer draws into a selected paint layer
 * or mask (as opaque mask coverage); any other selection, or Place on = new layer, adds a parametric shape layer.
 * Shift squares a box and Alt grows it from the press point, both read live on every move and the release, as the
 * marquee does; polygon and freehand ignore them. The platform's mod key (⌘ on macOS, where Ctrl-click opens the
 * context menu; Ctrl elsewhere) cuts the shape's opaque footprint out of the selected paint layer or mask instead,
 * whatever Place on says, and never adds a layer: the release decides for drags and traces, the closing press for a
 * click-closed polygon; Enter always draws. Pixel commits clip to the selection and bbox. Hidden, locked or loading
 * targets refuse with a notice (a cutout also refuses a transparency lock or a selection it cannot cut), at the
 * press when the press already knows, so nothing is drawn in vain; a refused polygon close, by click or Enter, keeps
 * its vertices. A new layer the document refuses (busy, not ready, over budget) is reported too.
 * Commits wait for the gesture to end, so a new layer is never refused mid-gesture: a drag or trace commits on
 * release, a polygon on Enter or on the release of the press that closes it. Escape (through the engine ladder) or a
 * cancelled press drops the session. A box or freehand gesture bound for the selected layer's pixels tells the engine
 * so as soon as its modifiers decide it, so a mask it edits shows above a staged preview; polygons never do. Previews
 * and cancellation do not dispatch, and a nonzero shape produces one structural or stroke commit.
 */

import type { CanvasLayerEditRefusal, CanvasShapeCutoutRefusal } from '@workbench/canvas-engine/capabilities';
import type { CanvasLayerSourceContract, CanvasRasterLayerContractV2 } from '@workbench/canvas-engine/contracts';
import type { SemanticLeaf } from '@workbench/canvas-engine/document-model/semanticLeaf';
import type { ShapeToolOptions } from '@workbench/canvas-engine/engineStores';
import type { CanvasProjectMutation } from '@workbench/canvas-engine/mutationContracts';
import type { PlacedSurface, PointerInput, PointerModifiers, Rect, Vec2 } from '@workbench/canvas-engine/types';

import { lookupDocumentLeaf } from '@workbench/canvas-engine/document-model/documentModel';
import { getDocumentLeaves } from '@workbench/canvas-engine/document/documentIndex';
import { isLayerTransparencyLocked, leafEditRefusal } from '@workbench/canvas-engine/document/layerEligibility';
import { isMaskLayer } from '@workbench/canvas-engine/document/sources';
import { polygonBounds } from '@workbench/canvas-engine/freehand';
import { identity, invert, multiply, translate } from '@workbench/canvas-engine/math/mat2d';
import { intersect, roundOut, transformBounds } from '@workbench/canvas-engine/math/rect';
import { drawShapeSource } from '@workbench/canvas-engine/render/rasterizers/shapeRasterizer';
import { IS_MAC_OS } from '@workbench/hotkeys/keys';

import type { Tool, ToolContext } from './tool';

import { dragRect, type DragRectConstraints } from './dragRect';
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
import { shapeDrawsOnLayer, shapePaintSourceOf } from './shapeTarget';

type ShapeSource = Extract<CanvasLayerSourceContract, { type: 'shape' }>;

/** Bit for the primary (usually left) mouse button in `PointerEvent.buttons`. */
const PRIMARY_BUTTON = 1;

/** Screen-space distance (CSS px) the pointer must travel before a press becomes a drag. */
export const SHAPE_DRAG_THRESHOLD_PX = 3;

/** Whether a finished shape adds its pixels (or a layer) or erases its footprint from the selected layer. */
type ShapeMode = 'draw' | 'cutout';

/** The platform's mod key cuts the shape out: ⌘ on macOS, where Ctrl-click opens the context menu; Ctrl elsewhere. */
const modeOf = (modifiers: PointerModifiers, isMacOs: boolean): ShapeMode =>
  (isMacOs ? modifiers.meta : modifiers.ctrl) ? 'cutout' : 'draw';

/** Shift squares a box drag and Alt centres it on the press point. */
const constraintsOf = (modifiers: PointerModifiers): DragRectConstraints => ({
  fromCenter: modifiers.alt,
  square: modifiers.shift,
});

/** Legacy's subtract cue: the crosshair with a minus badge, falling back to the plain crosshair. */
const CUTOUT_CURSOR_SVG = `<svg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24'>
  <g fill='none' stroke-linecap='round'>
    <path d='M12 3v5M12 16v5M3 12h5M16 12h5' stroke='white' stroke-width='3'/>
    <path d='M12 3v5M12 16v5M3 12h5M16 12h5' stroke='black' stroke-width='1.5'/>
    <circle cx='6' cy='18' r='4.5' fill='white' stroke='black' stroke-width='1'/>
    <path d='M4 18h4' stroke='black' stroke-width='1.5'/>
  </g>
</svg>`;

type Session =
  | { kind: 'drag'; startDoc: Vec2; startScreen: Vec2; moved: boolean; editsSelected: boolean }
  | {
      kind: 'polyline';
      session: PolylineSession;
      /** A press is down; a close it asks for waits for its release. */
      pressed: boolean;
      /** The close a press (or Enter during one) armed, committed on the release in this mode. */
      closing: ShapeMode | null;
    }
  | ({ kind: 'freehand'; editsSelected: boolean } & FreehandTrace);

const distance = (a: Vec2, b: Vec2): number => Math.hypot(a.x - b.x, a.y - b.y);

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

/** Where a finished shape goes: a new shape layer, the selected leaf's pixels, or nowhere, with the reason. */
type ShapeTarget =
  | { kind: 'layer' }
  | { kind: 'pixels'; leaf: SemanticLeaf }
  | { kind: 'refused'; refusal: CanvasLayerEditRefusal | CanvasShapeCutoutRefusal };

/** With fill off and no stroke to draw, a shape changes no pixel, so it records no step and uploads nothing. */
const drawsNothing = (source: ShapeSource): boolean =>
  source.fill === null && (source.stroke === null || source.strokeWidth <= 0);

/** The shape's footprint as coverage: each enabled part paints opaque, whatever its color. */
const coverageOf = (source: ShapeSource): ShapeSource => ({
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

/**
 * Where a shape finished in `mode` would go now. A cutout needs a selected paint layer or mask; a drawn shape takes
 * one only when Place on asks for it. `loading: 'request'` also refuses a target whose durable pixels are not cached
 * yet (drawing then would lose them) and asks for them; `'skip'` predicts without side effects, for the cursor.
 */
const resolveTarget = (ctx: ToolContext, mode: ShapeMode, loading: 'request' | 'skip'): ShapeTarget => {
  const doc = ctx.getDocument();
  const leaf = doc?.selectedLayerId ? lookupDocumentLeaf(doc, doc.selectedLayerId) : null;
  if (!leaf || !shapeDrawsOnLayer(leaf.layer)) {
    return mode === 'cutout' ? { kind: 'refused', refusal: { status: 'cutout-no-target' } } : { kind: 'layer' };
  }
  if (mode === 'draw' && ctx.stores.shapeOptions.get().target !== 'selected') {
    return { kind: 'layer' };
  }
  const { layer } = leaf;
  let status: CanvasLayerEditRefusal['status'] | null = leafEditRefusal(leaf);
  if (!status && mode === 'cutout' && isLayerTransparencyLocked(layer)) {
    status = 'layer-alpha-locked';
  }
  if (!status && loading === 'request' && shapePaintSourceOf(layer)?.bitmap) {
    const existing = ctx.layers.get(layer.id);
    if (!existing || existing.stale) {
      ctx.requestLayerRasterization?.(layer.id);
      status = 'layer-loading';
    }
  }
  return status
    ? { kind: 'refused', refusal: { layerId: layer.id, layerName: layer.name, status } }
    : { kind: 'pixels', leaf };
};

/** Creates a fresh shape tool with its own gesture state; `isMacOs` picks the cutout key. */
export const createShapeTool = ({ isMacOs = IS_MAC_OS }: { isMacOs?: boolean } = {}): Tool => {
  let session: Session | null = null;
  /** The cutout key on the latest pointer event, which picks the cursor cue. */
  let cutoutHeld = false;
  const cutoutCursor = `url("data:image/svg+xml,${encodeURIComponent(CUTOUT_CURSOR_SVG)}") 12 12, crosshair`;

  const clearPreview = (ctx: ToolContext): void => {
    ctx.stores.shapePreview.set(null);
    ctx.stores.lassoPreview.set(null);
    ctx.invalidate({ overlay: true });
  };

  const reset = (ctx: ToolContext): void => {
    session = null;
    clearPreview(ctx);
  };

  const noteModifiers = (ctx: ToolContext, modifiers: PointerModifiers): void => {
    const held = modeOf(modifiers, isMacOs) === 'cutout';
    if (held !== cutoutHeld) {
      cutoutHeld = held;
      ctx.updateCursor();
    }
  };

  /**
   * Tells the engine, once per box or freehand gesture, that it writes the selected layer's pixels as soon as the
   * modifiers make it so: at the press, or when the mod key turns a drag into a cutout. The engine asks
   * `editsSelectedLayer` before the press reaches the tool, too early to see its modifiers, so the tool answers here
   * instead. Polygons never count: vertex presses write nothing, and a close commits after its gesture has ended.
   */
  const noteSelectedLayerEdit = (
    ctx: ToolContext,
    gesture: Extract<Session, { editsSelected: boolean }>,
    modifiers: PointerModifiers
  ): void => {
    if (!gesture.editsSelected && resolveTarget(ctx, modeOf(modifiers, isMacOs), 'skip').kind === 'pixels') {
      gesture.editsSelected = true;
      ctx.beginSelectedLayerEdit?.();
    }
  };

  /** Reports why a press in `mode` would place nothing, so the user learns before drawing; true when refused. */
  const refusesPress = (ctx: ToolContext, mode: ShapeMode): boolean => {
    const target = resolveTarget(ctx, mode, 'request');
    if (target.kind !== 'refused') {
      return false;
    }
    ctx.reportEditRefusal?.(target.refusal);
    return true;
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
   * One pixel-shape stroke within selection/bbox clips, drawn through the layer inverse with the same masks as brush
   * painting, on paint pixels or mask alpha alike. A drawn shape grows the cache to cover it; a cutout erases its
   * opaque footprint from the pixels already cached and never grows it.
   */
  const commitPixels = (ctx: ToolContext, leaf: SemanticLeaf, placed: PlacedShape, mode: ShapeMode): void => {
    const layer = leaf.layer;
    if (drawsNothing(placed.source)) {
      return;
    }
    const toLocal = invert(layerMatrix(layer.transform));
    if (!toLocal) {
      // A zero-scale layer has no pixels to draw into.
      return;
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
    let dirtyRect = docRect ? roundOut(transformBounds(toLocal, docRect)) : null;
    const original = ctx.layers.peek(layer.id);
    if (dirtyRect && mode === 'cutout') {
      // Only cached pixels can be erased; an uncached layer has none.
      dirtyRect = original ? intersect(dirtyRect, original.rect) : null;
    }
    if (!dirtyRect || dirtyRect.width < 1 || dirtyRect.height < 1) {
      return;
    }
    // Admitted before the first pixel changes: the before/after pair is the step's footprint.
    const edit = ctx.beginStrokeEdit(dirtyRect.width * dirtyRect.height * 8);
    if (!edit) {
      return;
    }
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
      const source = mode === 'cutout' || isMaskLayer(layer) ? coverageOf(placed.source) : placed.source;
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
      surfaceCtx.globalCompositeOperation =
        mode === 'cutout' ? 'destination-out' : isLayerTransparencyLocked(layer) ? 'source-atop' : 'source-over';
      surfaceCtx.drawImage(scratch.canvas, sx, sy);
      surfaceCtx.restore();
      const afterImageData = surfaceCtx.getImageData(sx, sy, dirtyRect.width, dirtyRect.height);
      const tool = mode === 'cutout' ? 'shape-cutout' : 'shape';
      if (!edit.commit({ afterImageData, beforeImageData, dirtyRect, layerId: layer.id, tool })) {
        restore();
      }
    } catch (error) {
      restore();
      throw error;
    }
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
    const { status } = ctx.commitStructural('Add shape', forward, inverse);
    // Refusals before dispatch say why the release added nothing; failures after it were reported where they arose.
    if (status === 'busy' || status === 'gesture-active' || status === 'not-ready' || status === 'over-budget') {
      ctx.reportEditRefusal?.(status);
    }
  };

  /**
   * Places a finished shape in `mode` and ends the session. Called only once the gesture has ended. A refused selected
   * target places nothing, rather than adding a layer the user did not ask for, and says why; with `keepRefused` (a
   * polygon close) the session stays, so the shape can close again once the layer is fixed.
   */
  const commit = (ctx: ToolContext, placed: PlacedShape | null, mode: ShapeMode, keepRefused = false): void => {
    const target = placed && ctx.getDocument() ? resolveTarget(ctx, mode, 'request') : null;
    if (target?.kind === 'refused') {
      ctx.reportEditRefusal?.(target.refusal);
      if (keepRefused) {
        return;
      }
    }
    reset(ctx);
    if (!placed || !target) {
      return;
    }
    if (target.kind === 'pixels') {
      commitPixels(ctx, target.leaf, placed, mode);
    } else if (target.kind === 'layer') {
      commitLayer(ctx, placed);
    }
  };

  const boxShape = (ctx: ToolContext, rect: Rect): PlacedShape | null => {
    const kind = ctx.stores.shapeOptions.get().kind;
    if (rect.width < 1 || rect.height < 1 || kind === 'polygon' || kind === 'freehand') {
      return null;
    }
    return { rect, source: { ...styleFromOptions(ctx), height: rect.height, kind, type: 'shape', width: rect.width } };
  };

  /** Every polygon close, by click, double-click or Enter; a refused one keeps the vertices. */
  const closePolyline = (ctx: ToolContext, polyline: Extract<Session, { kind: 'polyline' }>, mode: ShapeMode): void => {
    polyline.closing = null;
    commit(ctx, polygonShapeFrom(polyline.session.points, styleFromOptions(ctx)), mode, true);
  };

  const pressPolygon = (ctx: ToolContext, input: PointerInput): void => {
    const mode = modeOf(input.modifiers, isMacOs);
    if (session?.kind !== 'polyline') {
      if (refusesPress(ctx, mode)) {
        return;
      }
      session = { closing: null, kind: 'polyline', pressed: true, session: startPolyline(input) };
    } else {
      session.pressed = true;
      if (pressPolyline(ctx, session.session, input) === 'close') {
        // A refused close keeps the vertices, so the user can fix the layer (or let go of Ctrl) and close again.
        if (!refusesPress(ctx, mode)) {
          session.closing = mode;
        }
        return;
      }
    }
    ctx.stores.lassoPreview.set(polylinePreview(session.session));
    ctx.invalidate({ overlay: true });
  };

  return {
    cursor: (ctx) => {
      if (session?.kind === 'polyline' && session.session.closeArmed) {
        return 'pointer';
      }
      const mode: ShapeMode = cutoutHeld ? 'cutout' : 'draw';
      if (resolveTarget(ctx, mode, 'skip').kind === 'refused') {
        return 'not-allowed';
      }
      return mode === 'cutout' ? cutoutCursor : 'crosshair';
    },
    id: 'shape',
    usesAltKey: true,
    onDeactivate: (ctx, opts) => {
      // A modifier-hold switch (space → view to pan mid-polygon) keeps the session.
      if (!opts?.temporary) {
        cutoutHeld = false;
        reset(ctx);
      }
    },
    onKeyCommand: (ctx, command) => {
      if (!session) {
        return false;
      }
      if (command === 'cancel') {
        reset(ctx);
        return true;
      }
      if (session.kind !== 'polyline') {
        return false;
      }
      if (session.pressed) {
        // Enter during a held press closes on its release, once the gesture has ended.
        session.closing ??= 'draw';
      } else {
        closePolyline(ctx, session, 'draw');
      }
      return true;
    },
    onPointerCancel: (ctx) => reset(ctx),
    onPointerDown: (ctx, input) => {
      if ((input.buttons & PRIMARY_BUTTON) === 0 || !ctx.getDocument()) {
        return;
      }
      noteModifiers(ctx, input.modifiers);
      const kind = ctx.stores.shapeOptions.get().kind;
      if (session && !sessionFitsKind(session, kind)) {
        reset(ctx);
      }
      if (kind === 'polygon') {
        pressPolygon(ctx, input);
        return;
      }
      if (session || refusesPress(ctx, modeOf(input.modifiers, isMacOs))) {
        return;
      }
      session =
        kind === 'freehand'
          ? { ...startFreehandTrace(input), editsSelected: false, kind: 'freehand' }
          : {
              editsSelected: false,
              kind: 'drag',
              moved: false,
              startDoc: input.documentPoint,
              startScreen: input.screenPoint,
            };
      noteSelectedLayerEdit(ctx, session, input.modifiers);
    },
    onPointerMove: (ctx, input, batch) => {
      noteModifiers(ctx, input.modifiers);
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
      noteSelectedLayerEdit(ctx, session, input.modifiers);
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
        const rect = dragRect(session.startDoc, input.documentPoint, constraintsOf(input.modifiers));
        ctx.stores.shapePreview.set({ kind, rect });
        ctx.invalidate({ overlay: true });
      }
    },
    onPointerUp: (ctx, input) => {
      noteModifiers(ctx, input.modifiers);
      if (!session) {
        return;
      }
      if (session.kind === 'polyline') {
        session.pressed = false;
        if (session.closing) {
          closePolyline(ctx, session, session.closing);
        }
        return;
      }
      const mode = modeOf(input.modifiers, isMacOs);
      if (session.kind === 'freehand') {
        commit(ctx, polygonShapeFrom(finishFreehandTrace(session, input), styleFromOptions(ctx)), mode);
        return;
      }
      if (!session.moved) {
        reset(ctx);
        return;
      }
      commit(ctx, boxShape(ctx, dragRect(session.startDoc, input.documentPoint, constraintsOf(input.modifiers))), mode);
    },
  };
};
