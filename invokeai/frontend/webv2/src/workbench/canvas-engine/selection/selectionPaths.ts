/** Pure document-space SVG path builders feed injected Path2D creation for selection commits and node tests. */

import type { Rect } from '@workbench/canvas-engine/types';

/** SVG path data for a closed rectangle. */
export const rectPathData = (r: Rect): string =>
  `M ${r.x} ${r.y} L ${r.x + r.width} ${r.y} L ${r.x + r.width} ${r.y + r.height} L ${r.x} ${r.y + r.height} Z`;

/** Exact ellipse from two half-arcs; a single closed arc would have coincident endpoints and be dropped. */
export const ellipsePathData = (r: Rect): string => {
  const rx = r.width / 2;
  const ry = r.height / 2;
  const cy = r.y + ry;
  const left = r.x;
  const right = r.x + r.width;
  return `M ${left} ${cy} A ${rx} ${ry} 0 1 0 ${right} ${cy} A ${rx} ${ry} 0 1 0 ${left} ${cy} Z`;
};

/** SVG path data for `kind` inscribed in `rect`. */
export const selectionShapePathData = (kind: 'rect' | 'ellipse', rect: Rect): string =>
  kind === 'ellipse' ? ellipsePathData(rect) : rectPathData(rect);
