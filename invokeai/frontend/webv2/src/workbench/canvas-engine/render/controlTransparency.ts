/**
 * Display-only control transparency matches legacy LightnessToAlphaFilter: alpha becomes min(existing alpha, (min
 * RGB + max RGB)/2), dropping dark backgrounds. Generation omits this effect and composites the control image at
 * full opacity.
 */

import type { RasterBackend, RasterSurface } from './raster';

/** In-place RGBA alpha = min(existing alpha, HSL lightness); RGB is unchanged. */
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

export const renderControlTransparency = (
  backend: RasterBackend,
  cache: RasterSurface,
  width: number,
  height: number,
  target: RasterSurface | null = null
): RasterSurface => {
  const out = target ?? backend.createSurface(width, height);
  if (out.width !== width || out.height !== height) {
    out.resize(width, height);
  }
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
