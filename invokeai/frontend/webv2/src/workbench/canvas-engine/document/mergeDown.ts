/**
 * Merge down preserves the lower transform, so map upper pixels through `inverse(lowerTransform) * upperTransform`
 * into lower-local space. Later lower-layer compositing restores the original document placement.
 */

import type { CanvasLayerBaseContract } from '@workbench/canvas-engine/contracts';
import type { Mat2d } from '@workbench/canvas-engine/types';

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
