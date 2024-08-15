import type { CanvasRasterLayerState } from 'features/controlLayers/store/types';

export const isValidLayer = (layer: CanvasRasterLayerState) => {
  return layer.isEnabled && layer.objects.length > 0;
};
