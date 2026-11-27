import type { CanvasEntityState, CanvasRectState } from 'features/controlLayers/store/types';

import { getTransparencyLockedCompositeOperation } from './transparencyLocking';

export const getShapeCompositeOperation = (
  entity: CanvasEntityState | null | undefined,
  isSubtracting: boolean
): CanvasRectState['compositeOperation'] => {
  if (isSubtracting) {
    return 'destination-out';
  }

  return getTransparencyLockedCompositeOperation(entity) === 'source-atop' ? 'source-atop' : 'source-over';
};
