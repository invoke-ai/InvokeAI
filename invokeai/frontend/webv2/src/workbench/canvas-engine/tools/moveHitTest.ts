/**
 * Transform document points into layer-local content bounds for exact rotation/scale hit tests. All renderable
 * source types and mask alpha use the same bounds; empty content returns null/false. Helpers locate a specified
 * layer, never choose a stack target instead of panel selection.
 */

import type {
  CanvasDocumentContractV3,
  CanvasLayerBaseContract,
  CanvasLayerContract,
} from '@workbench/canvas-engine/contracts';
import type { Rect, Vec2 } from '@workbench/canvas-engine/types';

import { getSourceContentRect, renderableSourceOf } from '@workbench/canvas-engine/document/sources';
import { applyToPoint, fromTRS, invert } from '@workbench/canvas-engine/math/mat2d';
import { isEmpty, union } from '@workbench/canvas-engine/math/rect';

type LayerTransform = CanvasLayerBaseContract['transform'];

/**
 * Local content bounds include off-origin paint/mask pixels and share {@link getSourceContentRect}. Empty or
 * unsupported layers return null.
 */
export const hittableLayerRect = (
  layer: CanvasLayerContract,
  doc: CanvasDocumentContractV3,
  liveRect?: Rect
): Rect | null => {
  if (!renderableSourceOf(layer)) {
    return null;
  }
  const content = getSourceContentRect(layer, doc);
  // Union live cache bounds so unflushed first strokes and newly extended content remain grabbable.
  const rect = liveRect && !isEmpty(liveRect) ? (isEmpty(content) ? liveRect : union(content, liveRect)) : content;
  if (rect.width <= 0 || rect.height <= 0) {
    return null;
  }
  return rect;
};

export const hittableLayerSize = (
  layer: CanvasLayerContract,
  doc: CanvasDocumentContractV3
): { width: number; height: number } | null => {
  const rect = hittableLayerRect(layer, doc);
  return rect ? { height: rect.height, width: rect.width } : null;
};

/** The layer's local→document affine matrix from its transform. */
export const layerMatrix = (transform: LayerTransform) =>
  fromTRS({ x: transform.x, y: transform.y }, transform.rotation, transform.scaleX, transform.scaleY);

/**
 * Tests a document-space point against rendered layer bounds.
 * @internal No production caller.
 */
export const hitTestLayer = (
  layer: CanvasLayerContract,
  doc: CanvasDocumentContractV3,
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
 * Returns transformed content corners or null. Partial overrides preview move/scale/rotation without mutation;
 * omitted fields retain committed values.
 */
export const layerOutlineCorners = (
  layer: CanvasLayerContract,
  doc: CanvasDocumentContractV3,
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
