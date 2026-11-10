import type { VectorTraceResult } from '../capabilities';
import type { CanvasDocumentContractV3 } from '../contracts';
import type { CanvasNodeInsertionAnchor } from '../document/insertionAnchors';
import type { RasterBackend } from '../render/raster';
import type { Rect, Vec2 } from '../types';
import type { CanvasMutationContext } from './mutationContext';

import { lookupDocumentLeaf } from '../document-model/documentModel';
import { approximateBezierPath } from '../math/bezierPath';
import { buildTaperedTracePoints, buildTraceLinePath, buildTraceOutlinePath, getTraceLineBounds } from '../math/vectorTrace';
import { findVectorFillRegionsAsync, VectorFillLimitError, VECTOR_FILL_LIMITS } from '../math/vectorFillRegions';
import { polygonBounds } from '../freehand';
import { roundOut, union } from '../math/rect';
import { getStroke } from 'perfect-freehand';
import { addedLayerHistoryBytes, paintLayerAt, publishAddedRasterLayer, rgbaBytes } from './editSteps';

type Options = {
  document: CanvasDocumentContractV3;
  layerId: string;
  pathId?: string;
  appearance?: boolean;
  fillOnly?: boolean;
  width: number;
  color: string;
  ctx: CanvasMutationContext;
  backend: RasterBackend;
  id: string;
  anchor: CanvasNodeInsertionAnchor;
  signal?: AbortSignal;
  isCurrent(): boolean;
  /** The controller can be tested against recording surfaces without browser globals. */
  createPath?: (data?: string) => Path2D;
};

/** Prepare against one immutable snapshot; admit pixels/history only after cancelable geometry work. */
export const traceVectorLayer = async (o: Options): Promise<VectorTraceResult> => {
  const current = () => !o.signal?.aborted && o.isCurrent();
  if (!current()) { return { status: o.signal?.aborted ? 'aborted' : 'stale' }; }
  if (!o.ctx.canEdit()) { return { status: 'busy' }; }
  if (o.ctx.isGestureActive()) { return { status: 'gesture-active' }; }
  const leaf = lookupDocumentLeaf(o.document, o.layerId);
  if (!leaf || leaf.layer.type !== 'vector' || !leaf.contributionEnabled || leaf.documentHidden || leaf.effectiveLocked) {
    return { status: 'not-ready' };
  }
  const layer = leaf.layer;
  const fill = o.fillOnly === true || (o.appearance === true && (layer.fillEnabled ?? false));
  const stroke = !o.fillOnly && (o.appearance !== true || (layer.strokeEnabled ?? true));
  if (!fill && !stroke) { return { status: 'empty' }; }
  const width = Math.max(1, Math.min(600, Math.round(o.width)));
  const paths = o.pathId === undefined ? layer.paths : layer.paths.filter((path) => path.id === o.pathId);
  const sampleCount = paths.reduce((sum, path) => sum + Math.max(0, path.points.length - (path.isClosed ? 0 : 1)) * 24, 0);
  if (sampleCount > (fill && !o.fillOnly ? VECTOR_FILL_LIMITS.segments : 200_000)) { return { status: 'too-complex' }; }
  const createPath = o.createPath ?? ((data?: string) => data ? new Path2D(data) : new Path2D());
  try {
    const traces: { coordinates: Vec2[]; closed: boolean; outline: number[][] | null; bounds: Rect }[] = [];
    for (const path of paths) {
      o.signal?.throwIfAborted();
      const coordinates = approximateBezierPath(path.points, path.isClosed);
      if (coordinates.length < 2) { continue; }
      const outline = stroke && layer.traceTaperEnds && !path.isClosed
        ? getStroke(buildTaperedTracePoints(coordinates, width, layer.traceTaper ?? 100), {
          size: width / 2, simulatePressure: false, last: true, thinning: 1,
        }) : null;
      const bounds = outline ? polygonBounds(outline.map((point) => ({ x: point[0]!, y: point[1]! })))
        : stroke ? getTraceLineBounds(coordinates) : polygonBounds(coordinates);
      traces.push({ coordinates, closed: path.isClosed, outline, bounds });
    }
    const regions = !fill ? [] : o.fillOnly
      ? traces.filter((trace) => trace.closed && trace.coordinates.length >= 3).map((trace) => trace.coordinates)
      : await findVectorFillRegionsAsync(traces.map((trace) => trace.coordinates), { signal: o.signal });
    if (!current()) { return { status: o.signal?.aborted ? 'aborted' : 'stale' }; }
    let bounds: Rect | null = null;
    for (const region of regions) { bounds = bounds ? union(bounds, polygonBounds(region)) : polygonBounds(region); }
    if (stroke) { for (const trace of traces) { bounds = bounds ? union(bounds, trace.bounds) : trace.bounds; } }
    if (!bounds) { return { status: 'empty' }; }
    // Control-point hulls bound all tension curves; actual pressure outlines bound taper. Include stroke + AA.
    const padding = stroke ? width / 2 + 2 : 2;
    const rect = roundOut({ x: bounds.x - padding, y: bounds.y - padding, width: bounds.width + padding * 2, height: bounds.height + padding * 2 });
    const txn = o.ctx.begin({ historyBytes: addedLayerHistoryBytes(rect) });
    if (!('publish' in txn)) { return txn; }
    try {
      if (!current()) { return { status: o.signal?.aborted ? 'aborted' : 'stale' }; }
      if (!txn.reserveRaster(rgbaBytes(rect))) { return { status: 'over-budget' }; }
      const pixels = o.backend.createSurface(rect.width, rect.height), draw = pixels.ctx;
      draw.setTransform(1, 0, 0, 1, -rect.x, -rect.y);
      draw.fillStyle = o.color; draw.strokeStyle = o.color;
      draw.lineWidth = width; draw.lineCap = 'round'; draw.lineJoin = 'round';
      if (regions.length > 0) {
        const regionPath = createPath();
        for (const region of regions) {
          regionPath.moveTo(region[0]!.x, region[0]!.y);
          for (const point of region.slice(1)) { regionPath.lineTo(point.x, point.y); }
          regionPath.closePath();
        }
        draw.fillStyle = layer.fillColor ?? '#ffffff'; draw.fill(regionPath); draw.fillStyle = o.color;
      }
      if (stroke) {
        for (const trace of traces) {
          if (trace.outline) { draw.fill(buildTraceOutlinePath(trace.outline, createPath)); }
          else { draw.stroke(buildTraceLinePath(trace.coordinates, createPath)); }
        }
      }
      const raster = paintLayerAt(o.id, `${layer.name} ${o.fillOnly ? 'Fill' : o.appearance ? 'Trace-Fill' : 'Trace'}`, rect);
      raster.transform = { ...layer.transform };
      const result = publishAddedRasterLayer(o.ctx, txn, {
        label: o.fillOnly ? 'Fill vector path' : o.appearance ? 'Trace/Fill vector paths' : 'Trace vector paths',
        layer: raster, rect, pixels, anchor: o.anchor, preserveSelection: true, selectedLayerId: o.document.selectedLayerId,
      });
      return { status: result.status === 'committed' ? 'committed' : result.status === 'over-budget' ? 'over-budget' : 'failed' };
    } finally { txn.end(); }
  } catch (error) {
    if (o.signal?.aborted) { return { status: 'aborted' }; }
    return { status: error instanceof VectorFillLimitError ? 'too-complex' : 'failed' };
  }
};
