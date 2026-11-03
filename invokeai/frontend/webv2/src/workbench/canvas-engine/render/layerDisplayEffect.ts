/**
 * Floating selections need the same raster adjustments/control transparency as their layer. Per-pixel effects are
 * region independent, so a display copy baked once at lift, at the cut's layer-local origin, matches the pixels it
 * left (noise included). Spatial adjustments are not: they depend on the whole layer the float will land in, so the
 * float previews the per-pixel part only and committing re-evaluates the layer. Transform changes do not affect it;
 * appearance changes cancel the float.
 */

import type { CanvasLayerContract } from '@workbench/canvas-engine/contracts';

import {
  type AdjustmentOrigin,
  applyAdjustments,
  isIdentityAdjustments,
  perPixelAdjustments,
} from '@workbench/canvas-engine/render/adjustments';
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
 * Returns an effect-baked copy, or null when drawing the source directly needs no allocation. `origin` is the
 * layer-local pixel of `source`'s top-left.
 */
export const renderLayerDisplayEffect = (
  backend: RasterBackend,
  layer: CanvasLayerContract,
  source: RasterSurface,
  origin: AdjustmentOrigin
): RasterSurface | null => {
  const adjustments = layer.type === 'raster' ? perPixelAdjustments(layer.adjustments) : undefined;
  const hasEffect = layer.type === 'raster' ? !isIdentityAdjustments(adjustments) : hasLayerDisplayEffect(layer);
  if (!hasEffect || source.width <= 0 || source.height <= 0) {
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
    applyAdjustments(imageData, adjustments, origin);
  } else {
    applyLightnessToAlpha(imageData.data);
  }
  ctx.putImageData(imageData, 0, 0);
  return out;
};
