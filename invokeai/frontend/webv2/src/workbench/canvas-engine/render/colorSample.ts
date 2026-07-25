/**
 * Samples the composited document color at a document-space point, for the
 * color-picker tool. Composites just the renderable layers (bottom → top,
 * respecting opacity/blend/transform, matching {@link compositeDocument}'s
 * paint order) into a 1×1 scratch surface translated so the sampled pixel
 * lands at its origin — cheap regardless of document size, since only one
 * destination pixel is ever rasterized.
 *
 * The document background (solid fill or checkerboard) and the staged
 * preview are intentionally excluded: a point with no layer coverage samples
 * as fully transparent (`null`), not the page chrome.
 *
 * Zero React, zero import-time side effects.
 */

import type { CanvasDocumentContractV2 } from '@workbench/canvas-engine/contracts';
import type { Mat2d, Vec2 } from '@workbench/canvas-engine/types';

import type { LayerCacheStore } from './layerCache';
import type { RasterBackend } from './raster';

import { compositeDocument } from './compositor';

/** An RGBA sample, channels in `[0, 255]`. */
export interface RgbaSample {
  r: number;
  g: number;
  b: number;
  a: number;
}

/**
 * Samples the composited document color at `docPoint` (document space;
 * fractional coordinates are floored to the covering pixel). The canvas plane
 * is unbounded, so transformed layer content outside the document dimensions
 * remains pickable. Returns `null` when no renderable layer covers the point
 * with non-zero alpha.
 */
export const sampleDocumentColor = (
  doc: CanvasDocumentContractV2,
  layers: LayerCacheStore,
  backend: RasterBackend,
  docPoint: Vec2
): RgbaSample | null => {
  const px = Math.floor(docPoint.x);
  const py = Math.floor(docPoint.y);
  if (!Number.isFinite(px) || !Number.isFinite(py)) {
    return null;
  }

  const scratch = backend.createSurface(1, 1);
  const view: Mat2d = { a: 1, b: 0, c: 0, d: 1, e: -px, f: -py };
  // Reuse the canonical compositor so sampling shares its layer ordering,
  // cache-origin placement, transforms, blend modes, and display effects.
  // Omitting a checkerboard tile and staged preview keeps empty space transparent.
  compositeDocument(scratch, doc, layers, view, { backend });

  const { data } = scratch.ctx.getImageData(0, 0, 1, 1);
  const alpha = data[3] ?? 0;
  if (alpha === 0) {
    return null;
  }
  return { a: alpha, b: data[2] ?? 0, g: data[1] ?? 0, r: data[0] ?? 0 };
};
