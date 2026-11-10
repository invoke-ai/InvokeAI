import type { CanvasDocumentContractV3 } from '../contracts';
import type { CanvasNodeInsertionAnchor } from '../document/insertionAnchors';
import type { RasterBackend } from '../render/raster';
import type { CanvasMutationContext } from './mutationContext';

import { lookupDocumentLeaf } from '../document-model/documentModel';
import { approximateBezierPath } from '../math/bezierPath';
import { buildTaperedTracePoints, buildTraceLinePath, buildTraceOutlinePath } from '../math/vectorTrace';
import { polygonBounds } from '../freehand';
import { roundOut } from '../math/rect';
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
};

/** Trace all paths, or one exact path, using the same layer settings and pixel-history seam. */
export const traceVectorLayer = (o: Options): boolean => {
  const leaf = lookupDocumentLeaf(o.document, o.layerId);
  if (!leaf || leaf.layer.type !== 'vector' || !leaf.contributionEnabled || leaf.documentHidden || leaf.effectiveLocked) { return false; }
  const layer = leaf.layer;
  const fill = o.fillOnly === true || (o.appearance === true && (layer.fillEnabled ?? false));
  const stroke = !o.fillOnly && (o.appearance !== true || (layer.strokeEnabled ?? true));
  if (!fill && !stroke) { return false; }
  const width = Math.max(1, Math.min(600, Math.round(o.width)));
  const paths = o.pathId === undefined ? layer.paths : layer.paths.filter((path) => path.id === o.pathId);
  const traces = paths.filter((path) => stroke || (fill && path.isClosed)).map((path) => ({ coordinates: approximateBezierPath(path.points, path.isClosed), closed: path.isClosed }))
    .filter((trace) => trace.coordinates.length >= (stroke ? 2 : 3));
  if (traces.length === 0) { return false; }
  const bounds = polygonBounds(traces.flatMap((trace) => trace.coordinates));
  const padding = stroke ? width / 2 + 2 : 2;
  const rect = roundOut({ x: bounds.x - padding, y: bounds.y - padding, width: bounds.width + padding * 2, height: bounds.height + padding * 2 });
  const txn = o.ctx.begin({ historyBytes: addedLayerHistoryBytes(rect) });
  if (!('publish' in txn)) { return false; }
  try {
    if (!txn.reserveRaster(rgbaBytes(rect))) { return false; }
    const pixels = o.backend.createSurface(rect.width, rect.height);
    const draw = pixels.ctx;
    draw.translate(-rect.x, -rect.y);
    draw.fillStyle = o.color;
    draw.strokeStyle = o.color;
    draw.lineWidth = width;
    draw.lineCap = 'round'; draw.lineJoin = 'round';
    for (const trace of traces) {
      if (fill && trace.closed) {
        draw.beginPath();
        draw.moveTo(trace.coordinates[0]!.x, trace.coordinates[0]!.y);
        for (const point of trace.coordinates.slice(1)) { draw.lineTo(point.x, point.y); }
        draw.closePath();
        draw.fillStyle = layer.fillColor ?? '#ffffff';
        draw.fill();
        draw.fillStyle = o.color;
      }
      if (!stroke) { continue; }
      draw.beginPath();
      if (layer.traceTaperEnds && !trace.closed) {
        const outline = getStroke(buildTaperedTracePoints(trace.coordinates, width, layer.traceTaper ?? 100), {
          size: width / 2, simulatePressure: false, last: true, thinning: 1,
        });
        draw.fill(buildTraceOutlinePath(outline));
      } else {
        // v6 traces a sampled closed path as an open brush line whose last point repeats the first.
        draw.stroke(buildTraceLinePath(trace.coordinates));
      }
    }
    const raster = paintLayerAt(o.id, `${layer.name} ${o.fillOnly ? 'Fill' : o.appearance ? 'Trace-Fill' : 'Trace'}`, rect);
    raster.transform = { ...layer.transform };
    return publishAddedRasterLayer(o.ctx, txn, { label: o.fillOnly ? 'Fill vector path' : o.appearance ? 'Trace/Fill vector paths' : 'Trace vector paths', layer: raster, rect, pixels,
      anchor: o.anchor, preserveSelection: true, selectedLayerId: o.document.selectedLayerId }).status === 'committed';
  } finally { txn.end(); }
};
