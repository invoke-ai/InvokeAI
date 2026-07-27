import type { CanvasEntityState, CompositeOperation } from 'features/controlLayers/store/types';

export const getTransparencyLockedCompositeOperation = (
  entity: CanvasEntityState | null | undefined
): CompositeOperation | undefined => {
  if (entity?.type === 'raster_layer' && entity.isTransparencyLocked) {
    return 'source-atop';
  }

  return undefined;
};

export const getDrawingCompositeOperation = (
  entity: CanvasEntityState | null | undefined,
  fallback: CompositeOperation = 'source-over'
): CompositeOperation => {
  return getTransparencyLockedCompositeOperation(entity) ?? fallback;
};
