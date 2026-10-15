import type { CanvasLayerSourceContract } from '@workbench/canvas-engine/contracts';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';

import { isEmptyPolygonShape } from '@workbench/canvas-engine/document/sources';

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
export { rasterizeTextSource, textFontString, textFontVariationSettings } from './textRasterizer';

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
      if (isEmptyPolygonShape(source)) {
        throw new Error('rasterizeSource: a polygon shape needs at least three points');
      }
      return rasterizeShapeSource(source, deps, target);
    case 'gradient':
      return rasterizeGradientSource(source, deps, target);
    case 'text':
      return rasterizeTextSource(source, deps, target);
  }
};
