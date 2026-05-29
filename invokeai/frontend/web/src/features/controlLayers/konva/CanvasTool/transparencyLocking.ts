import type { CanvasEntityState, CompositeOperation } from 'features/controlLayers/store/types';

export const getTransparencyLockedCompositeOperation = (
  entity: CanvasEntityState | null | undefined
): CompositeOperation | undefined => {
  if (entity?.type === 'raster_layer' && entity.isTransparencyLocked) {
    return 'source-atop';
  }

  return undefined;
};
