/**
 * The control-layer transparency effect (legacy `LightnessToAlphaFilter`).
 *
 * A control layer with `withTransparencyEffect` renders its source so DARK areas
 * become transparent — for an edge/depth/pose map, the black background drops out
 * and the underlying content shows through. This mirrors legacy
 * `features/controlLayers/konva/filters.ts`: per pixel, HSL lightness
 * `(min(r,g,b) + max(r,g,b)) / 2` becomes the new alpha, clamped to never EXCEED
 * the pixel's existing alpha.
 *
 * It is DISPLAY-ONLY: at generation time the control image is composited at full
 * opacity with NO transparency effect (see `generation/canvas/compositePlan.ts`
 * `toControlLayerRef`), exactly like legacy (which rasterizes control with
 * `filters: []`, `bg: 'black'`).
 *
 * Zero React, zero import-time side effects.
 */

import type { RasterBackend, RasterSurface } from './raster';

/**
 * Applies the lightness→alpha transform to an RGBA pixel buffer in place: each
 * pixel's alpha becomes `min(alpha, lightness)` where `lightness` is the HSL
 * lightness of its RGB. Pure — the unit-testable core of the effect.
 */
export const applyLightnessToAlpha = (data: Uint8ClampedArray): void => {
  for (let i = 0; i + 3 < data.length; i += 4) {
    const r = data[i] ?? 0;
    const g = data[i + 1] ?? 0;
    const b = data[i + 2] ?? 0;
    const a = data[i + 3] ?? 0;
    const lightness = (Math.min(r, g, b) + Math.max(r, g, b)) / 2;
    data[i + 3] = Math.min(a, lightness);
  }
};

/**
 * Produces a transparency-effect copy of a control layer's cache surface (same
 * dimensions): the cache is drawn, its pixels transformed by
 * {@link applyLightnessToAlpha}, and the result returned. The caller blits it
 * through the layer transform in place of the raw cache.
 */
export const renderControlTransparency = (
  backend: RasterBackend,
  cache: RasterSurface,
  width: number,
  height: number
): RasterSurface => {
  const out = backend.createSurface(width, height);
  const ctx = out.ctx;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = 'source-over';
  ctx.drawImage(cache.canvas, 0, 0);
  const imageData = ctx.getImageData(0, 0, width, height);
  applyLightnessToAlpha(imageData.data);
  ctx.putImageData(imageData, 0, 0);
  return out;
};
