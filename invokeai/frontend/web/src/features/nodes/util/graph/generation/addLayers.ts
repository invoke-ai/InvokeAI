import type { CanvasLayerState } from 'features/controlLayers/store/types';

export const isValidLayer = (entity: CanvasLayerState) => {
  return (
    entity.isEnabled &&
    // Boolean(entity.bbox) && TODO(psyche): Re-enable this check when we have a way to calculate bbox for all layers
    entity.objects.length > 0
  );
};
