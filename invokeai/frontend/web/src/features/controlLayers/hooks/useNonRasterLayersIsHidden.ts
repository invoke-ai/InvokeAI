import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import { useMemo } from 'react';

export const useNonRasterLayersIsHidden = (): boolean => {
  const selectNonRasterLayersIsHidden = useMemo(
    () =>
      createSelector(selectCanvasSlice, (canvas) => {
        // Check if all non-raster layer categories are hidden
        return (
          canvas.controlLayers.isHidden &&
          canvas.inpaintMasks.isHidden &&
          canvas.regionalGuidance.isHidden
        );
      }),
    []
  );
  const isHidden = useAppSelector(selectNonRasterLayersIsHidden);
  return isHidden;
};