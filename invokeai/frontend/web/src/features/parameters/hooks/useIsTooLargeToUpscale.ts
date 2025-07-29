import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import type { ImageWithDims } from 'features/controlLayers/store/types';
import { selectUpscaleSlice } from 'features/parameters/store/upscaleSlice';
import { selectConfigSlice } from 'features/system/store/configSlice';
import { useMemo } from 'react';

const createIsTooLargeToUpscaleSelector = (imageWithDims?: ImageWithDims | null) =>
  createSelector(selectUpscaleSlice, selectConfigSlice, (upscale, config) => {
    const { upscaleModel, scale } = upscale;
    const { maxUpscaleDimension } = config;

    if (!maxUpscaleDimension || !upscaleModel || !imageWithDims) {
      // When these are missing, another warning will be shown
      return false;
    }

    const { width, height } = imageWithDims;

    const maxPixels = maxUpscaleDimension ** 2;
    const upscaledPixels = width * scale * height * scale;

    return upscaledPixels > maxPixels;
  });

export const useIsTooLargeToUpscale = (imageWithDims?: ImageWithDims | null) => {
  const selectIsTooLargeToUpscale = useMemo(() => createIsTooLargeToUpscaleSelector(imageWithDims), [imageWithDims]);
  return useAppSelector(selectIsTooLargeToUpscale);
};
