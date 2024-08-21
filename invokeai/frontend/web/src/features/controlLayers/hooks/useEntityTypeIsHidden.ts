import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectCanvasV2Slice } from 'features/controlLayers/store/canvasV2Slice';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { useMemo } from 'react';

export const useEntityTypeIsHidden = (type: CanvasEntityIdentifier['type']): boolean => {
  const selectIsHidden = useMemo(
    () =>
      createSelector(selectCanvasV2Slice, (canvasV2) => {
        switch (type) {
          case 'control_layer':
            return canvasV2.controlLayers.isHidden;
          case 'raster_layer':
            return canvasV2.rasterLayers.isHidden;
          case 'inpaint_mask':
            return canvasV2.inpaintMasks.isHidden;
          case 'regional_guidance':
            return canvasV2.regions.isHidden;
          case 'ip_adapter':
          default:
            return false;
        }
      }),
    [type]
  );
  const isHidden = useAppSelector(selectIsHidden);
  return isHidden;
};
