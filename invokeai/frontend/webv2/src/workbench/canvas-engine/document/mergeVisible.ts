import type { CanvasLayerContract } from '@workbench/canvas-engine/contracts';

export type HasMergeVisibleContent = (layerId: string) => boolean;

/** Returns eligible contributors in document order (top-most first). */
export const getMergeVisibleRasterLayers = (
  layers: readonly CanvasLayerContract[],
  hasContent: HasMergeVisibleContent
): CanvasLayerContract[] =>
  layers.filter((layer) => layer.type === 'raster' && layer.isEnabled && hasContent(layer.id));

/** Whether the raster group's merge-visible action has at least two contributors. */
export const canMergeVisibleRasters = (
  layers: readonly CanvasLayerContract[],
  hasContent: HasMergeVisibleContent
): boolean => getMergeVisibleRasterLayers(layers, hasContent).length >= 2;
