/**
 * Pure geometry for the "fit bbox to layers / masks" header actions, mirroring
 * legacy `CanvasBboxToolModule.fitToLayers` / `useAutoFitBBoxToMasks`:
 *
 * 1. Union the DOCUMENT-space bounds of the qualifying ENABLED, renderable,
 *    non-empty layers (all layers for fit-to-layers; INPAINT masks only for
 *    fit-to-masks — legacy fits `getVisibleRectOfType('inpaint_mask')`, so
 *    regional-guidance masks are deliberately excluded).
 * 2. For masks, expand the union outward by a fixed padding (legacy adds
 *    `maskBlur + 8`; webv2 does not plumb the mask-blur param through the widget
 *    yet, so a constant {@link MASK_FIT_PADDING} is used — see the report).
 * 3. Snap the rect INWARD to the model bbox grid via {@link fitRectToGrid} so the
 *    result lands on the same multiples a manual bbox edit snaps to.
 *
 * Returns `null` when there is nothing to fit (no qualifying content, or the
 * snapped rect collapses to empty) so the caller can disable the button and never
 * dispatch a degenerate bbox. React feeds the grid size from generate settings
 * (`gridSizeForModelBase`), exactly like the bbox tool.
 *
 * Zero React, zero engine/DOM state — unit-tested in node.
 */

import type { Rect } from '@workbench/canvas-engine/types';
import type { CanvasDocumentContractV2, CanvasLayerContract } from '@workbench/types';

import { getSourceBounds, isRenderableLayer } from '@workbench/canvas-engine/document/sources';
import { isEmpty, union } from '@workbench/canvas-engine/math/rect';

/** Fixed outward padding (document px) applied to the mask union before grid-fitting. */
const MASK_FIT_PADDING = 8;

/**
 * Snaps a rect INWARD to `gridSize`: the top-left edges round up (toward the
 * interior) and the width/height round down, so the result aligns to the grid and
 * never exceeds the input rect. `gridSize <= 1` degrades to a plain integer round
 * (no snapping — used when snap-to-grid is off). Mirrors legacy `fitRectToGrid`.
 */
export const fitRectToGrid = (rect: Rect, gridSize: number): Rect => {
  const g = gridSize > 1 ? gridSize : 1;
  const x = Math.ceil(rect.x / g) * g;
  const y = Math.ceil(rect.y / g) * g;
  const width = Math.floor((rect.width - (x - rect.x)) / g) * g;
  const height = Math.floor((rect.height - (y - rect.y)) / g) * g;
  return { height, width, x, y };
};

/**
 * The union of the document-space bounds of every ENABLED, renderable layer whose
 * content is non-empty and that satisfies `predicate`, or `null` when none
 * qualify. Empty layers (a brand-new bitmap-less mask, an empty paint layer)
 * contribute a zero-size rect and are skipped, matching legacy's `hasObjects()`
 * gate.
 */
export const unionRenderableBounds = (
  doc: CanvasDocumentContractV2,
  predicate: (layer: CanvasLayerContract) => boolean
): Rect | null => {
  let bounds: Rect | null = null;
  for (const layer of doc.layers) {
    if (!isRenderableLayer(layer) || !predicate(layer)) {
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
export const computeFitBboxToLayers = (doc: CanvasDocumentContractV2, gridSize: number): Rect | null => {
  const bounds = unionRenderableBounds(doc, () => true);
  if (!bounds) {
    return null;
  }
  const fitted = fitRectToGrid(bounds, gridSize);
  return isEmpty(fitted) ? null : fitted;
};

/** The padded, grid-snapped bbox that fits the visible INPAINT masks (legacy parity), or `null` when there are none. */
export const computeFitBboxToMasks = (
  doc: CanvasDocumentContractV2,
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
