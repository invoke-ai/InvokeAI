import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { useMemo } from 'react';

export const useEntityTypeCount = (type: CanvasEntityIdentifier['type']): number => {
  const selectEntityCount = useMemo(
    () =>
      createSelector(selectCanvasSlice, (canvas) => {
        switch (type) {
          case 'control_layer':
            return canvas.controlLayers.entities.length;
          case 'raster_layer':
            return canvas.rasterLayers.entities.length;
          case 'inpaint_mask':
            return canvas.inpaintMasks.entities.length;
          case 'regional_guidance':
            return canvas.regionalGuidance.entities.length;
          case 'reference_image':
            return canvas.referenceImages.entities.length;
          default:
            return 0;
        }
      }),
    [type]
  );
  const entityCount = useAppSelector(selectEntityCount);
  return entityCount;
};
