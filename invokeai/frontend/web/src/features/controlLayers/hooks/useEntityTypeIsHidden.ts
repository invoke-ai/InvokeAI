import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { useMemo } from 'react';

export const useEntityTypeIsHidden = (type: CanvasEntityIdentifier['type']): boolean => {
  const selectIsHidden = useMemo(
    () =>
      createSelector(selectCanvasSlice, (canvas) => {
        switch (type) {
          case 'control_layer':
            return canvas.controlLayers.isHidden;
          case 'raster_layer':
            return canvas.rasterLayers.isHidden;
          case 'inpaint_mask':
            return canvas.inpaintMasks.isHidden;
          case 'regional_guidance':
            return canvas.regionalGuidance.isHidden;
          case 'reference_image':
          default:
            return false;
        }
      }),
    [type]
  );
  const isHidden = useAppSelector(selectIsHidden);
  return isHidden;
};
