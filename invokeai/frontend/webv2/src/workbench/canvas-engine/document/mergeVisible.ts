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

/**
 * Destructive merge-selected may only collapse one uninterrupted span of the
 * raster stack. Other layer groups do not participate in raster compositing
 * order, so a mask/control layer between two selected rasters is harmless; an
 * unselected raster between them is not.
 */
export const areSelectedRasterLayersContiguous = (
  layers: readonly CanvasLayerContract[],
  selectedLayerIds: ReadonlySet<string>
): boolean => {
  const rasterLayers = layers.filter((layer) => layer.type === 'raster');
  const selectedIndexes = rasterLayers.flatMap((layer, index) => (selectedLayerIds.has(layer.id) ? [index] : []));
  if (selectedIndexes.length < 2 || selectedIndexes.length !== selectedLayerIds.size) {
    return false;
  }
  return selectedIndexes.at(-1)! - selectedIndexes[0]! + 1 === selectedIndexes.length;
};

/** Whether merge-selected can flatten the exact selection without changing the rendered composite. */
export const canMergeSelectedRasters = (
  layers: readonly CanvasLayerContract[],
  selectedLayerIds: ReadonlySet<string>,
  hasContent: HasMergeVisibleContent
): boolean =>
  areSelectedRasterLayersContiguous(layers, selectedLayerIds) &&
  layers
    .filter((layer) => selectedLayerIds.has(layer.id))
    .every(
      (layer) =>
        layer.type === 'raster' &&
        layer.isEnabled &&
        !layer.isLocked &&
        layer.blendMode === 'normal' &&
        hasContent(layer.id)
    );
