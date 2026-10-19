/**
 * Floating selections need the same raster adjustments/control transparency as their layer. These per-pixel
 * effects commute with cutting, so bake a display copy once at lift. Transform changes do not affect it;
 * appearance changes cancel the float.
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

/** Returns an effect-baked copy, or null when drawing the source directly needs no allocation. */
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
