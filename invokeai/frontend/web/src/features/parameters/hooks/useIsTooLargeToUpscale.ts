import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { selectUpscaleSlice } from 'features/parameters/store/upscaleSlice';
import { selectConfigSlice } from 'features/system/store/configSlice';
import { useMemo } from 'react';
import type { ImageDTO } from 'services/api/types';

const createIsTooLargeToUpscaleSelector = (imageDTO?: ImageDTO | null) =>
  createMemoizedSelector(selectUpscaleSlice, selectConfigSlice, (upscale, config) => {
    const { upscaleModel, scale } = upscale;
    const { maxUpscaleDimension } = config;

    if (!maxUpscaleDimension || !upscaleModel || !imageDTO) {
      // When these are missing, another warning will be shown
      return false;
    }

    const { width, height } = imageDTO;

    const maxPixels = maxUpscaleDimension ** 2;
    const upscaledPixels = width * scale * height * scale;

    return upscaledPixels > maxPixels;
  });

export const useIsTooLargeToUpscale = (imageDTO?: ImageDTO | null) => {
  const selectIsTooLargeToUpscale = useMemo(() => createIsTooLargeToUpscaleSelector(imageDTO), [imageDTO]);
  return useAppSelector(selectIsTooLargeToUpscale);
};
