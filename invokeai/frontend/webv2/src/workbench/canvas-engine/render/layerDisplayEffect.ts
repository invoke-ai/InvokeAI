/**
 * The display-only effects a layer's pixels pass through on the way to screen:
 * raster **adjustments** (brightness / contrast / saturation / curves) and the
 * control-layer **transparency effect** (lightness→alpha, so a dark control map
 * drops its background out).
 *
 * The compositor applies these to a layer's cache every frame through memoizing
 * caches. This module exists for the one case that has pixels of its own outside
 * that cache: a floating selection. Its pixels were cut from the layer, so they
 * must render exactly as the layer's own pixels do — otherwise floating a region
 * of a control map shows the raw, opaque black background the effect normally
 * removes.
 *
 * Both effects are strictly PER-PIXEL, so applying one to a cut-out region gives
 * exactly the same result as applying it to the whole layer and then cutting.
 * That is what lets the float bake its display copy ONCE at lift time rather
 * than re-deriving it every frame: nothing about its transform can change the
 * outcome, and a change to the layer's display properties cancels the float.
 *
 * Zero React, zero import-time side effects.
 */

import type { CanvasLayerContract } from '@workbench/canvas-engine/contracts';

import { applyAdjustments, isIdentityAdjustments } from '@workbench/canvas-engine/render/adjustments';
import { applyLightnessToAlpha } from '@workbench/canvas-engine/render/controlTransparency';

import type { RasterBackend, RasterSurface } from './raster';

/** Whether `layer` renders its pixels through any display-only transform. */
export const hasLayerDisplayEffect = (layer: CanvasLayerContract): boolean => {
  if (layer.type === 'control') {
    return layer.withTransparencyEffect;
  }
  return layer.type === 'raster' && !isIdentityAdjustments(layer.adjustments);
};

/**
 * A copy of `source` with `layer`'s display effects baked in, or `null` when the
 * layer has none — the caller then draws `source` directly, which is the common
 * case and allocates nothing.
 */
export const renderLayerDisplayEffect = (
  backend: RasterBackend,
  layer: CanvasLayerContract,
  source: RasterSurface
): RasterSurface | null => {
  if (!hasLayerDisplayEffect(layer) || source.width <= 0 || source.height <= 0) {
    return null;
  }
  const out = backend.createSurface(source.width, source.height);
  const ctx = out.ctx;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = 'source-over';
  ctx.clearRect(0, 0, source.width, source.height);
  ctx.drawImage(source.canvas, 0, 0);

  const imageData = ctx.getImageData(0, 0, source.width, source.height);
  if (layer.type === 'raster') {
    applyAdjustments(imageData, layer.adjustments);
  } else {
    applyLightnessToAlpha(imageData.data);
  }
  ctx.putImageData(imageData, 0, 0);
  return out;
};
