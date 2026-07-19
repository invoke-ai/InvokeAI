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

import { isRenderableLayer } from '@workbench/canvas-engine/document/sources';
import { fromTRS, multiply } from '@workbench/canvas-engine/math/mat2d';

import type { LayerCacheStore } from './layerCache';
import type { RasterBackend } from './raster';

import { blendToComposite } from './compositor';

/** An RGBA sample, channels in `[0, 255]`. */
export interface RgbaSample {
  r: number;
  g: number;
  b: number;
  a: number;
}

const matToTuple = (m: Mat2d): [number, number, number, number, number, number] => [m.a, m.b, m.c, m.d, m.e, m.f];

/**
 * Samples the composited document color at `docPoint` (document space;
 * fractional coordinates are floored to the covering pixel). Returns `null`
 * when the point falls outside the document, or when no renderable layer
 * covers it with non-zero alpha there.
 */
export const sampleDocumentColor = (
  doc: CanvasDocumentContractV2,
  layers: LayerCacheStore,
  backend: RasterBackend,
  docPoint: Vec2
): RgbaSample | null => {
  const px = Math.floor(docPoint.x);
  const py = Math.floor(docPoint.y);
  if (px < 0 || py < 0 || px >= doc.width || py >= doc.height) {
    return null;
  }

  const scratch = backend.createSurface(1, 1);
  const ctx = scratch.ctx;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, 1, 1);
  // Translate so the sampled document pixel lands at the scratch surface's
  // one pixel (0, 0); each layer's own transform composes on top of this.
  const view: Mat2d = { a: 1, b: 0, c: 0, d: 1, e: -px, f: -py };

  // Bottom → top, matching the real compositor's paint order.
  for (let i = doc.layers.length - 1; i >= 0; i--) {
    const layer = doc.layers[i];
    if (!layer || !isRenderableLayer(layer)) {
      continue;
    }
    const entry = layers.get(layer.id);
    if (!entry) {
      continue;
    }

    ctx.save();
    ctx.globalAlpha = layer.opacity;
    ctx.globalCompositeOperation = blendToComposite(layer.blendMode);
    const layerMat = fromTRS(
      { x: layer.transform.x, y: layer.transform.y },
      layer.transform.rotation,
      layer.transform.scaleX,
      layer.transform.scaleY
    );
    ctx.setTransform(...matToTuple(multiply(view, layerMat)));
    ctx.drawImage(entry.surface.canvas, 0, 0);
    ctx.restore();
  }

  const { data } = ctx.getImageData(0, 0, 1, 1);
  const alpha = data[3] ?? 0;
  if (alpha === 0) {
    return null;
  }
  return { a: alpha, b: data[2] ?? 0, g: data[1] ?? 0, r: data[0] ?? 0 };
};
