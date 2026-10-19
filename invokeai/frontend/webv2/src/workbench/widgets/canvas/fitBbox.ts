/**
 * Union enabled, renderable, nonempty layer bounds in document space. Mask fitting includes only inpaint masks and
 * adds {@link MASK_FIT_PADDING}; snap inward with {@link fitRectToGrid}. Return null for no content or a collapsed
 * result.
 */

import {
  compileDocumentLeaves,
  getSourceBounds,
  isEmpty,
  isRenderableLayer,
  type CanvasDocumentContractV3,
  type CanvasLayerContract,
  type Rect,
  union,
} from '@workbench/canvas-engine/api';

/** Fixed outward padding (document px) applied to the mask union before grid-fitting. */
const MASK_FIT_PADDING = 8;

/**
 * Snap inward: round origins up and sizes down. gridSize <= 1 uses integer rounding, matching legacy
 * fitRectToGrid.
 */
export const fitRectToGrid = (rect: Rect, gridSize: number): Rect => {
  const g = gridSize > 1 ? gridSize : 1;
  const x = Math.ceil(rect.x / g) * g;
  const y = Math.ceil(rect.y / g) * g;
  const width = Math.floor((rect.width - (x - rect.x)) / g) * g;
  const height = Math.floor((rect.height - (y - rect.y)) / g) * g;
  return { height, width, x, y };
};

/** Union document bounds of enabled, renderable, nonempty layers matching predicate; return null when none qualify. */
export const unionRenderableBounds = (
  doc: CanvasDocumentContractV3,
  predicate: (layer: CanvasLayerContract) => boolean
): Rect | null => {
  let bounds: Rect | null = null;
  for (const leaf of compileDocumentLeaves(doc)) {
    const layer = leaf.layer;
    if (!leaf.contributionEnabled || !isRenderableLayer(layer) || !predicate(layer)) {
      continue;
    }
    const layerBounds = getSourceBounds(layer, doc);
    if (isEmpty(layerBounds)) {
      continue;
    }
    bounds = bounds ? union(bounds, layerBounds) : layerBounds;
  }
  return bounds;
};

/** The grid-snapped bbox that tightly fits all visible content, or `null` when there is none. */
export const computeFitBboxToLayers = (doc: CanvasDocumentContractV3, gridSize: number): Rect | null => {
  const bounds = unionRenderableBounds(doc, () => true);
  if (!bounds) {
    return null;
  }
  const fitted = fitRectToGrid(bounds, gridSize);
  return isEmpty(fitted) ? null : fitted;
};

/** The padded, grid-snapped bbox that fits the visible INPAINT masks (legacy parity), or `null` when there are none. */
export const computeFitBboxToMasks = (
  doc: CanvasDocumentContractV3,
  gridSize: number,
  padding: number = MASK_FIT_PADDING
): Rect | null => {
  const bounds = unionRenderableBounds(doc, (layer) => layer.type === 'inpaint_mask');
  if (!bounds) {
    return null;
  }
  const expanded: Rect = {
    height: bounds.height + padding * 2,
    width: bounds.width + padding * 2,
    x: bounds.x - padding,
    y: bounds.y - padding,
  };
  const fitted = fitRectToGrid(expanded, gridSize);
  return isEmpty(fitted) ? null : fitted;
};
