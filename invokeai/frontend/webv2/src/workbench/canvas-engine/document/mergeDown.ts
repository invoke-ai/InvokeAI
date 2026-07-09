/**
 * Pure transform math for merging one layer down into the layer below it.
 *
 * "Merge down" bakes the upper layer's pixels into the lower layer while the
 * reducer keeps the lower layer's transform verbatim (see
 * `mergeCanvasLayersDown` — it never touches pixels). So the engine must draw
 * the upper layer's cache into the lower layer's *local* space: upper-local →
 * document (via the upper transform) → lower-local (via the inverse of the
 * lower transform). Composing those gives the matrix returned here, which the
 * engine feeds to `ctx.setTransform` when blitting the upper cache onto the
 * merged surface. When the merged (lower) layer is later composited with its
 * unchanged transform, the upper content lands back at its original document
 * position: `lowerTransform · (lowerTransform⁻¹ · upperTransform) = upperTransform`.
 *
 * Zero React, zero DOM, zero import-time side effects.
 */

import type { Mat2d } from '@workbench/canvas-engine/types';
import type { CanvasLayerBaseContract } from '@workbench/types';

import { fromTRS, invert, multiply } from '@workbench/canvas-engine/math/mat2d';

type LayerTransform = CanvasLayerBaseContract['transform'];

const transformToMat = (t: LayerTransform): Mat2d => fromTRS({ x: t.x, y: t.y }, t.rotation, t.scaleX, t.scaleY);

/**
 * The matrix that maps the upper layer's local (cache) space into the lower
 * layer's local space: `inverse(lower) · upper`. Returns `null` when the lower
 * transform is singular (zero scale) and cannot be inverted.
 */
export const mergeDownMatrix = (lowerTransform: LayerTransform, upperTransform: LayerTransform): Mat2d | null => {
  const lowerInverse = invert(transformToMat(lowerTransform));
  if (!lowerInverse) {
    return null;
  }
  return multiply(lowerInverse, transformToMat(upperTransform));
};
