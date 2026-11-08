/**
 * Which layers take a finished shape as pixels. A leaf module, so the properties pane can ask without loading the
 * shape tool or its rasterizer.
 */

import type { CanvasLayerContract, CanvasLayerSourceContract } from '@workbench/canvas-engine/contracts';

import { maskAsPaintSource } from '@workbench/canvas-engine/document/sources';

/**
 * The paint pixels a shape draws into or cuts out of: a raster paint layer's source, or a mask's alpha viewed as
 * paint. Null for every other layer, which a drawn shape lands beside as a new shape layer instead.
 */
export const shapePaintSourceOf = (
  layer: CanvasLayerContract
): Extract<CanvasLayerSourceContract, { type: 'paint' }> | null =>
  layer.type === 'raster' && layer.source.type === 'paint' ? layer.source : maskAsPaintSource(layer);

/**
 * Whether the shape tool draws onto `layer` itself (as paint pixels or mask coverage) when it is the selected layer,
 * whatever its locks or visibility: those refuse the shape rather than redirect it.
 */
export const shapeDrawsOnLayer = (layer: CanvasLayerContract): boolean => shapePaintSourceOf(layer) !== null;
