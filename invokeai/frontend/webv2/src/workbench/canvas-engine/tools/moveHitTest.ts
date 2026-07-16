/**
 * Pure geometry for the move tool: layer hit-testing and outline corners.
 *
 * A layer's raster cache holds unscaled pixels in the layer's LOCAL space; the
 * compositor draws it through the layer transform (translate·rotate·scale). To
 * hit-test a document-space point we invert that transform and check the point
 * against the layer's native `[0,w]×[0,h]` rect — so rotation and scale are
 * handled exactly, not via an axis-aligned approximation.
 *
 * Every renderable layer is hit-testable — image by its native size, paint by
 * its content rect (bitmap dims at the persisted offset), the parametric sources
 * (shape/gradient/text) by the same content rect the compositor/cache use
 * (CANVAS_PLAN Phase 5: "param for parametric"), and MASK layers (inpaint /
 * regional) by their alpha bitmap's content rect (legacy allows moving masks).
 * An EMPTY layer (a brand-new / cleared paint or mask layer with no pixels)
 * returns `null`/`false` rather than throwing, so the move/transform tools can
 * scan a mixed stack safely.
 *
 * Zero React, zero import-time side effects.
 */

import type { Rect, Vec2 } from '@workbench/canvas-engine/types';
import type { CanvasDocumentContractV2, CanvasLayerBaseContract, CanvasLayerContract } from '@workbench/types';

import {
  getSourceContentRect,
  LAYER_GROUP_COUNT,
  layerGroupRank,
  renderableSourceOf,
} from '@workbench/canvas-engine/document/sources';
import { applyToPoint, fromTRS, invert } from '@workbench/canvas-engine/math/mat2d';
import { isEmpty, union } from '@workbench/canvas-engine/math/rect';

type LayerTransform = CanvasLayerBaseContract['transform'];

/**
 * Resolves a layer's LIVE cache rect (layer-local), or `undefined`. Threaded from
 * the engine so hit-testing covers freshly-painted content the debounced bitmap
 * flush hasn't yet written back to the persisted contract — mirrors how
 * `invertMask` unions the live cache rect. A pure caller (no engine) omits it.
 */
export type LiveCacheRectOf = (layerId: string) => Rect | undefined;

/**
 * A hit-testable layer's content rectangle in its LOCAL (untransformed) space —
 * including its (possibly off-zero) origin for content-sized paint/mask layers —
 * or `null` for a layer with no rasterizable content or an EMPTY layer (no
 * pixels). Mask layers hit-test by their alpha bitmap's content rect, so the move
 * tool can grab a painted mask (matching legacy). Every source (image/paint/
 * shape/gradient/text) and mask reuses {@link getSourceContentRect} — the exact
 * rect the cache/compositor render.
 */
export const hittableLayerRect = (
  layer: CanvasLayerContract,
  doc: CanvasDocumentContractV2,
  liveRect?: Rect
): Rect | null => {
  if (!renderableSourceOf(layer)) {
    return null;
  }
  const content = getSourceContentRect(layer, doc);
  // Union the live cache rect (when present): a stroke painted moments ago lives
  // in the cache but not yet in the persisted contract (the bitmap flush is
  // debounced), so content alone would leave freshly-painted pixels ungrabbable —
  // and content is EMPTY for a brand-new paint layer whose first stroke hasn't
  // flushed. Mirrors `invertMask`'s live-cache union.
  const rect = liveRect && !isEmpty(liveRect) ? (isEmpty(content) ? liveRect : union(content, liveRect)) : content;
  if (rect.width <= 0 || rect.height <= 0) {
    return null;
  }
  return rect;
};

/**
 * The native (unscaled) local pixel size of a hit-testable layer, or `null` when
 * not hit-testable. A thin wrapper over {@link hittableLayerRect} for callers
 * that only need eligibility / dimensions (not the origin).
 */
export const hittableLayerSize = (
  layer: CanvasLayerContract,
  doc: CanvasDocumentContractV2
): { width: number; height: number } | null => {
  const rect = hittableLayerRect(layer, doc);
  return rect ? { height: rect.height, width: rect.width } : null;
};

/** The layer's local→document affine matrix from its transform. */
export const layerMatrix = (transform: LayerTransform) =>
  fromTRS({ x: transform.x, y: transform.y }, transform.rotation, transform.scaleX, transform.scaleY);

/** True when `point` (document space) falls inside `layer`'s rendered bounds. */
export const hitTestLayer = (
  layer: CanvasLayerContract,
  doc: CanvasDocumentContractV2,
  point: Vec2,
  liveRect?: Rect
): boolean => {
  const rect = hittableLayerRect(layer, doc, liveRect);
  if (!rect) {
    return false;
  }
  const inverse = invert(layerMatrix(layer.transform));
  if (!inverse) {
    return false;
  }
  const local = applyToPoint(inverse, point);
  return local.x >= rect.x && local.x <= rect.x + rect.width && local.y >= rect.y && local.y <= rect.y + rect.height;
};

/**
 * The top-most layer that both satisfies `predicate` and contains `point`, or
 * `null`. "Top-most" follows the COMPOSITE order, not the raw array order: layers
 * are visited by group rank high→low (inpaint mask > regional guidance > control >
 * raster — see {@link layerGroupRank}), and WITHIN a group by array order (index 0
 * is top-most within its group). This matches what the compositor draws, so a
 * control layer painted visually over a raster wins the hit even when the raster
 * sits at a lower global index — keeping the move-tool auto-select and the canvas
 * context-menu target consistent with the pixels on screen (batch finding N1). The
 * predicate lets callers require visible/unlocked; `liveRectOf` (optional) supplies
 * each layer's live cache rect so freshly-painted, not-yet-flushed content is
 * hit-testable.
 */
export const topLayerAt = (
  doc: CanvasDocumentContractV2,
  point: Vec2,
  predicate: (layer: CanvasLayerContract) => boolean,
  liveRectOf?: LiveCacheRectOf
): CanvasLayerContract | null => {
  for (let rank = LAYER_GROUP_COUNT - 1; rank >= 0; rank--) {
    for (const layer of doc.layers) {
      if (layerGroupRank(layer) !== rank) {
        continue;
      }
      if (predicate(layer) && hitTestLayer(layer, doc, point, liveRectOf?.(layer.id))) {
        return layer;
      }
    }
  }
  return null;
};

/**
 * The four document-space corners of a layer's rendered rectangle (for the move
 * overlay outline), or `null` when the layer has no hit-testable bounds.
 * `override` shifts/scales/rotates the rect for a live drag preview without
 * mutating the layer (fields absent from the override fall back to the committed
 * transform; the move tool passes only `x`/`y`).
 */
export const layerOutlineCorners = (
  layer: CanvasLayerContract,
  doc: CanvasDocumentContractV2,
  override?: { x: number; y: number; scaleX?: number; scaleY?: number; rotation?: number } | null
): Vec2[] | null => {
  const rect = hittableLayerRect(layer, doc);
  if (!rect) {
    return null;
  }
  const transform: LayerTransform = override
    ? {
        rotation: override.rotation ?? layer.transform.rotation,
        scaleX: override.scaleX ?? layer.transform.scaleX,
        scaleY: override.scaleY ?? layer.transform.scaleY,
        x: override.x,
        y: override.y,
      }
    : layer.transform;
  const matrix = layerMatrix(transform);
  return [
    { x: rect.x, y: rect.y },
    { x: rect.x + rect.width, y: rect.y },
    { x: rect.x + rect.width, y: rect.y + rect.height },
    { x: rect.x, y: rect.y + rect.height },
  ].map((corner) => applyToPoint(matrix, corner));
};
