import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectCanvasV2Slice } from 'features/controlLayers/store/canvasV2Slice';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { useMemo } from 'react';

export const useEntityCount = (type: CanvasEntityIdentifier['type']): number => {
  const selectEntityCount = useMemo(
    () =>
      createSelector(selectCanvasV2Slice, (canvasV2) => {
        switch (type) {
          case 'control_layer':
            return canvasV2.controlLayers.entities.length;
          case 'raster_layer':
            return canvasV2.rasterLayers.entities.length;
          case 'inpaint_mask':
            return canvasV2.inpaintMasks.entities.length;
          case 'regional_guidance':
            return canvasV2.regions.entities.length;
          case 'ip_adapter':
            return canvasV2.ipAdapters.entities.length;
          default:
            return 0;
        }
      }),
    [type]
  );
  const entityCount = useAppSelector(selectEntityCount);
  return entityCount;
};
