/**
 * Pure contributor selection for the raster group's "merge visible" action.
 *
 * Legacy merge-visible composites every visible raster entity with content into
 * a new raster layer. Source kind and lock state do not affect participation:
 * locks prevent editing a source, not reading its rendered pixels.
 */

import type { CanvasLayerContract } from '@workbench/types';

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
