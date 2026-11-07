import type { CanvasBezierPointContract, CanvasVectorLayerContract, ParametricShapeKind } from '@workbench/canvas-engine/contracts';
import type { ShapeToolOptions } from '@workbench/canvas-engine/engineStores';
import type { Mat2d, PointerInput, Rect, Vec2 } from '@workbench/canvas-engine/types';

import { lookupDocumentLeaf } from '@workbench/canvas-engine/document-model/documentModel';
import { getDocumentIndex } from '@workbench/canvas-engine/document/documentIndex';
import { leafEditRefusal } from '@workbench/canvas-engine/document/layerEligibility';
import { createVectorLayer, nextVectorLayerName } from '@workbench/canvas-engine/document/layerFactories';
import { anchorsToBezierPoints, fitPolylineToBezierPoints, ovalToBezierPoints, rectToBezierPoints } from '@workbench/canvas-engine/math/bezierPath';
import { applyToPoint, invert } from '@workbench/canvas-engine/math/mat2d';
import { polygonShapeVertices } from '@workbench/canvas-engine/math/shapeGeometry';
import { getStrokePoints } from 'perfect-freehand';

import type { Tool, ToolContext } from './tool';

import { layerMatrix } from './moveHitTest';

type Session = {
  layer: CanvasVectorLayerContract;
  toLocal: Mat2d;
  kind: ShapeToolOptions['kind'];
  target: ShapeToolOptions['target'];
  start: Vec2;
  points: Vec2[];
  input: PointerInput;
  lastSample: Vec2;
  lastPress: { screen: Vec2; time: number };
  closeArmed: boolean;
  spaceHeld: boolean;
  finish?: boolean;
};

export interface VectorShapeTool extends Tool {
  hasSession(): boolean;
}

const distance = (a: Vec2, b: Vec2): number => Math.hypot(a.x - b.x, a.y - b.y);
const CLOSE_RADIUS_PX = 10;
const isBoxKind = (kind: ShapeToolOptions['kind']): kind is ParametricShapeKind =>
  kind !== 'polygon' && kind !== 'freehand';

const boxPoints = (kind: ParametricShapeKind, rect: Rect): CanvasBezierPointContract[] => {
  if (kind === 'rect') {
    return rectToBezierPoints(rect);
  }
  if (kind === 'ellipse') {
    return ovalToBezierPoints(rect);
  }
  return anchorsToBezierPoints(polygonShapeVertices(kind, rect));
};

/** v6 box gestures: Shift constrains aspect, Alt draws from center, Space translates the live box. */
const dragRect = (session: Session): Rect => {
  let dx = session.input.documentPoint.x - session.start.x;
  let dy = session.input.documentPoint.y - session.start.y;
  if (session.input.modifiers.shift) {
    const side = Math.max(Math.abs(dx), Math.abs(dy));
    const signX = dx !== 0 ? Math.sign(dx) : dy !== 0 ? Math.sign(dy) : 1;
    const signY = dy !== 0 ? Math.sign(dy) : signX;
    dx = side * signX;
    dy = side * signY;
  }
  const fromCenter = session.input.modifiers.alt;
  const x1 = session.start.x - (fromCenter ? dx : 0);
  const y1 = session.start.y - (fromCenter ? dy : 0);
  const x2 = session.start.x + dx;
  const y2 = session.start.y + dy;
  return { x: Math.min(x1, x2), y: Math.min(y1, y2), width: Math.abs(x2 - x1), height: Math.abs(y2 - y1) };
};

export const createVectorShapeTool = (): VectorShapeTool => {
  let session: Session | null = null;
  let cancelPreviewFrame: (() => void) | null = null;

  const reset = (ctx: ToolContext): void => {
    cancelPreviewFrame?.();
    cancelPreviewFrame = null;
    session = null;
    ctx.stores.vectorShapePreview.set(null);
    ctx.invalidate({ overlay: true });
  };

  const eligible = (ctx: ToolContext): boolean => {
    const doc = ctx.getDocument();
    const leaf = doc?.selectedLayerId ? lookupDocumentLeaf(doc, doc.selectedLayerId) : null;
    return !!leaf && leaf.layer.type === 'vector' && !leaf.documentHidden && leafEditRefusal(leaf) === null;
  };

  const isCurrent = (ctx: ToolContext): boolean => {
    if (!session) {
      return false;
    }
    const doc = ctx.getDocument();
    const leaf = doc?.selectedLayerId ? lookupDocumentLeaf(doc, doc.selectedLayerId) : null;
    const options = ctx.stores.shapeOptions.get();
    if (!leaf || leaf.layer.type !== 'vector' || leaf.id !== session.layer.id || leaf.documentHidden || leafEditRefusal(leaf) ||
      leaf.layer.paths !== session.layer.paths || leaf.layer.transform !== session.layer.transform ||
      options.kind !== session.kind || options.target !== session.target) {
      reset(ctx);
      return false;
    }
    return true;
  };

  const snapInput = (input: PointerInput, current: Session): PointerInput => {
    if (current.kind === 'freehand') {
      return input;
    }
    const local = applyToPoint(current.toLocal, input.documentPoint);
    const documentPoint = applyToPoint(layerMatrix(current.layer.transform), { x: Math.floor(local.x), y: Math.floor(local.y) });
    return { ...input, documentPoint };
  };

  const polygonCursor = (ctx: ToolContext, current: Session): Vec2 => {
    const point = current.input.documentPoint;
    const first = current.points[0]!;
    current.closeArmed = current.points.length >= 3 &&
      distance(ctx.viewport.documentToScreen(first), current.input.screenPoint) <= CLOSE_RADIUS_PX;
    if (current.closeArmed) {
      return first;
    }
    if (!current.input.modifiers.shift) {
      return point;
    }
    const last = current.points.at(-1)!;
    const length = distance(last, point);
    const angle = Math.round(Math.atan2(point.y - last.y, point.x - last.x) / (Math.PI / 4)) * Math.PI / 4;
    const snapped = { x: last.x + Math.cos(angle) * length, y: last.y + Math.sin(angle) * length };
    if (current.points.length >= 2) {
      const threshold = CLOSE_RADIUS_PX / ctx.viewport.getZoom();
      const dx = Math.abs(snapped.x - first.x);
      const dy = Math.abs(snapped.y - first.y);
      if (dx <= threshold && (dy > threshold || dx <= dy)) {
        snapped.x = first.x;
      } else if (dy <= threshold) {
        snapped.y = first.y;
      }
    }
    return snapped;
  };

  const preview = (ctx: ToolContext): void => {
    if (!session) {
      return;
    }
    const current = session;
    let points: CanvasBezierPointContract[];
    let isClosed = false;
    if (isBoxKind(current.kind)) {
      const rect = dragRect(current);
      points = rect.width > 0 && rect.height > 0
        ? boxPoints(current.kind, rect)
        : [];
      isClosed = true;
    } else if (current.kind === 'polygon') {
      const cursor = polygonCursor(ctx, current);
      const anchors = current.closeArmed || distance(cursor, current.points.at(-1)!) === 0
        ? current.points : [...current.points, cursor];
      points = anchorsToBezierPoints(anchors);
      isClosed = current.closeArmed;
    } else {
      points = anchorsToBezierPoints(current.points);
    }
    ctx.stores.vectorShapePreview.set({
      layerId: current.layer.id, points, isClosed,
      vertices: current.kind === 'polygon' ? current.points.slice() : null,
      closeRadiusPx: current.kind === 'polygon' && current.points.length >= 3 ? CLOSE_RADIUS_PX : null,
      closeArmed: current.closeArmed,
    });
    ctx.invalidate({ overlay: true });
    ctx.updateCursor();
  };

  // Build the growing freehand preview once per presented frame, not once per coalesced input event.
  const requestPreview = (ctx: ToolContext): void => {
    if (cancelPreviewFrame) {
      return;
    }
    let ran = false;
    const cancel = ctx.scheduleFrame(() => {
      ran = true;
      cancelPreviewFrame = null;
      preview(ctx);
    });
    if (!ran) {
      cancelPreviewFrame = cancel;
    }
  };

  const commit = (ctx: ToolContext, points: CanvasBezierPointContract[], isClosed: boolean): void => {
    if (!isCurrent(ctx) || !session) {
      return;
    }
    const current = session;
    const doc = ctx.getDocument()!;
    reset(ctx);
    if (points.length < (isClosed ? 3 : 2)) {
      return;
    }
    const toLocal = current.target === 'selected' ? current.toLocal : null;
    const path = {
      id: ctx.createLayerId(), name: null, isClosed,
      points: toLocal ? points.map((point) => ({
        ...point,
        anchor: applyToPoint(toLocal, point.anchor),
        inHandle: point.inHandle ? applyToPoint(toLocal, point.inHandle) : null,
        outHandle: point.outHandle ? applyToPoint(toLocal, point.outHandle) : null,
      })) : points,
    };
    if (current.target === 'new') {
      const layer = createVectorLayer({
        id: ctx.createLayerId(),
        name: nextVectorLayerName(getDocumentIndex(doc).nodes.map((entry) => entry.node.name)),
      });
      layer.paths = [path];
      ctx.commitStructural('Add vector shape', {
        type: 'addCanvasLayer', layer, anchor: ctx.captureInsertionAnchor('vector', current.layer.id),
      }, { type: 'removeCanvasLayers', ids: [layer.id] });
    } else {
      if (ctx.appendVectorEditPath?.(current.layer.id, path)) {
        return;
      }
      ctx.commitStructural('Add vector path', {
        type: 'updateCanvasLayerConfig', id: current.layer.id,
        config: { layerType: 'vector', paths: [...current.layer.paths, path] },
      }, {
        type: 'updateCanvasLayerConfig', id: current.layer.id,
        config: { layerType: 'vector', paths: current.layer.paths },
      });
    }
  };

  return {
    id: 'shape',
    hasSession: () => session !== null,
    cursor: (ctx) => {
      if (!eligible(ctx)) {
        return 'not-allowed';
      }
      if (session?.spaceHeld) {
        return isBoxKind(session.kind) ? 'move' : 'grabbing';
      }
      return session?.closeArmed ? 'pointer' : 'crosshair';
    },
    onDocumentChanged: (ctx) => {
      if (session) {
        isCurrent(ctx);
      }
    },
    onDeactivate: (ctx, options) => {
      if (!options?.temporary) {
        reset(ctx);
      }
    },
    onPointerCancel: reset,
    onModifierKey: (ctx, code, pressed) => {
      if (!session) {
        return code === 'Alt' && eligible(ctx) && isBoxKind(ctx.stores.shapeOptions.get().kind);
      }
      if (!isCurrent(ctx) || !session) {
        return false;
      }
      if (code === 'Space') {
        session.spaceHeld = pressed;
        session.input = snapInput({
          ...session.input,
          documentPoint: ctx.viewport.screenToDocument(session.input.screenPoint),
        }, session);
      } else {
        session.input = { ...session.input, modifiers: { ...session.input.modifiers, [code === 'Alt' ? 'alt' : 'shift']: pressed } };
      }
      requestPreview(ctx);
      return code === 'Space' || code === 'Alt' && isBoxKind(session.kind);
    },
    onKeyCommand: (ctx, command) => {
      if (command === 'cancel') {
        reset(ctx);
      } else if (isCurrent(ctx) && session?.kind === 'polygon') {
        commit(ctx, anchorsToBezierPoints(session.points), false);
      }
    },
    onPointerDown: (ctx, rawInput) => {
      if ((rawInput.buttons & 1) === 0 || !eligible(ctx)) {
        return;
      }
      if (session && !isCurrent(ctx)) {
        return;
      }
      if (!session) {
        const doc = ctx.getDocument()!;
        const layer = lookupDocumentLeaf(doc, doc.selectedLayerId!)!.layer as CanvasVectorLayerContract;
        const toLocal = invert(layerMatrix(layer.transform));
        if (!toLocal) {
          return;
        }
        const options = ctx.stores.shapeOptions.get();
        session = {
          layer, toLocal, kind: options.kind, target: options.target,
          start: rawInput.documentPoint, points: [], input: rawInput, lastSample: rawInput.screenPoint,
          lastPress: { screen: rawInput.screenPoint, time: -Infinity }, closeArmed: false, spaceHeld: false,
        };
        const input = snapInput(rawInput, session);
        session.input = input;
        session.start = input.documentPoint;
        if (session.kind === 'freehand') {
          session.points = [input.documentPoint];
        }
      }
      if (session.spaceHeld) {
        return;
      }
      if (session.kind === 'polygon') {
        session.input = snapInput(rawInput, session);
        const point = session.points.length > 0 ? polygonCursor(ctx, session) : session.input.documentPoint;
        if (session.closeArmed) {
          session.finish = true;
          requestPreview(ctx);
          return;
        }
        if (rawInput.timeStamp - session.lastPress.time <= 350 && distance(rawInput.screenPoint, session.lastPress.screen) <= 5) {
          session.finish = false;
          return;
        }
        session.points.push(point);
        session.lastPress = { screen: rawInput.screenPoint, time: rawInput.timeStamp };
      }
      requestPreview(ctx);
    },
    onPointerMove: (ctx, rawInput, batch) => {
      if (!isCurrent(ctx) || !session) {
        return;
      }
      const current = session;
      if (current.spaceHeld) {
        if (isBoxKind(current.kind)) {
          const input = snapInput(rawInput, current);
          current.start = {
            x: current.start.x + input.documentPoint.x - current.input.documentPoint.x,
            y: current.start.y + input.documentPoint.y - current.input.documentPoint.y,
          };
          current.input = input;
        } else {
          ctx.viewport.panBy({ x: rawInput.screenPoint.x - current.input.screenPoint.x, y: rawInput.screenPoint.y - current.input.screenPoint.y });
          current.input = { ...rawInput, documentPoint: ctx.viewport.screenToDocument(rawInput.screenPoint) };
          ctx.invalidate({ view: true, overlay: true });
        }
      } else {
        current.input = snapInput(rawInput, current);
        if (current.kind === 'freehand' && (rawInput.buttons & 1) !== 0) {
          for (const input of batch) {
            if (distance(input.screenPoint, current.lastSample) < 1) {
              continue;
            }
            const from = current.points.at(-1)!;
            const steps = Math.max(1, Math.ceil(distance(from, input.documentPoint) * ctx.viewport.getZoom() / 2));
            for (let step = 1; step <= steps; step += 1) {
              const t = step / steps;
              current.points.push({ x: from.x + (input.documentPoint.x - from.x) * t, y: from.y + (input.documentPoint.y - from.y) * t });
            }
            current.lastSample = input.screenPoint;
          }
        }
      }
      requestPreview(ctx);
    },
    onPointerUp: (ctx, rawInput) => {
      if (!isCurrent(ctx) || !session) {
        return;
      }
      const current = session;
      if (current.kind === 'polygon') {
        if (current.finish !== undefined) {
          commit(ctx, anchorsToBezierPoints(current.points), current.finish);
        }
        return;
      }
      if (current.kind === 'freehand') {
        if (current.points.length < 2) {
          reset(ctx);
          return;
        }
        const zoom = ctx.viewport.getZoom();
        const stabilized = getStrokePoints(current.points, {
          size: 1 / zoom, smoothing: 0.5, streamline: 0.35, thinning: 0, simulatePressure: false, last: true,
        }).map(({ point }) => ({ x: point[0]!, y: point[1]! }));
        commit(ctx, fitPolylineToBezierPoints(stabilized, Math.max(0.25, 2 / zoom)), false);
      } else {
        current.input = snapInput(rawInput, current);
        const rect = dragRect(current);
        if (rect.width <= 0 || rect.height <= 0) {
          reset(ctx);
          return;
        }
        commit(ctx, boxPoints(current.kind, rect), true);
      }
    },
  };
};
