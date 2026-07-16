/**
 * Source rasterizer dispatch.
 *
 * Given a layer `source`, produces a {@link RasterSurface} holding its
 * rasterized pixels. `image`, `paint`, `shape`, `gradient`, and `text` are
 * implemented. A `polygon` shape (no points-editing UX yet) throws — the
 * dispatch only routes rect/ellipse.
 *
 * Zero React, zero import-time side effects.
 */

import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { CanvasLayerSourceContract } from '@workbench/types';

import type { RasterizeDeps, RasterizeResult } from './types';

import { rasterizeGradientSource } from './gradientRasterizer';
import { rasterizeImageSource } from './imageRasterizer';
import { rasterizePaintSource } from './paintRasterizer';
import { rasterizeShapeSource } from './shapeRasterizer';
import { rasterizeTextSource } from './textRasterizer';

export type { ImageResolver, RasterizeDeps, RasterizeResult } from './types';
export { rasterizeGradientSource } from './gradientRasterizer';
export { rasterizeImageSource } from './imageRasterizer';
export { rasterizePaintSource } from './paintRasterizer';
export { rasterizeShapeSource } from './shapeRasterizer';
export { rasterizeTextSource } from './textRasterizer';

/**
 * Rasterizes any supported layer source into a surface. Throws only for the
 * deferred `polygon` shape kind (no rasterizer yet).
 */
export const rasterizeSource = (
  source: CanvasLayerSourceContract,
  deps: RasterizeDeps,
  target?: RasterSurface
): Promise<RasterizeResult> => {
  switch (source.type) {
    case 'image':
      return rasterizeImageSource(source, deps, target);
    case 'paint':
      return rasterizePaintSource(source, deps, target);
    case 'shape':
      if (source.kind === 'polygon') {
        throw new Error("rasterizeSource: 'polygon' shapes are not implemented yet (deferred)");
      }
      return rasterizeShapeSource(source, deps, target);
    case 'gradient':
      return rasterizeGradientSource(source, deps, target);
    case 'text':
      return rasterizeTextSource(source, deps, target);
  }
};
