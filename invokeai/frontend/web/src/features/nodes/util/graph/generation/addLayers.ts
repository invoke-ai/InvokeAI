import type { CanvasRasterLayerState } from 'features/controlLayers/store/types';

export const isValidLayerWithoutControlAdapter = (layer: CanvasRasterLayerState) => {
  return (
    layer.isEnabled &&
    // Boolean(entity.bbox) && TODO(psyche): Re-enable this check when we have a way to calculate bbox for all layers
    layer.objects.length > 0 &&
    layer.controlAdapter === null
  );
};
