/**
 * SVG path data for the primitive selection shapes.
 *
 * Selection geometry reaches the engine as a closed `Path2D` (see
 * {@link SelectionCommit}), built through the injected {@link CreatePath2D} seam
 * so it stays node-safe. These builders produce the path DATA — the string — so
 * they are pure, trivially testable, and usable on the node raster stub where
 * `Path2D` is a fake.
 *
 * All coordinates are DOCUMENT space, matching the space the selection mask is
 * placed in.
 *
 * Zero React, zero import-time side effects.
 */

import type { Rect } from '@workbench/canvas-engine/types';

/** SVG path data for a closed rectangle. */
export const rectPathData = (r: Rect): string =>
  `M ${r.x} ${r.y} L ${r.x + r.width} ${r.y} L ${r.x + r.width} ${r.y + r.height} L ${r.x} ${r.y + r.height} Z`;

/**
 * SVG path data for a closed ellipse inscribed in `r`, as two half-arcs from the
 * left extreme to the right and back. `A` (elliptical arc) is used rather than a
 * Bézier approximation so the curve is exact; a full ellipse cannot be a single
 * arc segment (start and end would coincide and the arc would be dropped), hence
 * the pair.
 */
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
