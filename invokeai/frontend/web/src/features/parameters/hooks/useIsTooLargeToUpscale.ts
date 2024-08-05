import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { selectUpscalelice } from 'features/parameters/store/upscaleSlice';
import { selectConfigSlice } from 'features/system/store/configSlice';
import { useMemo } from 'react';
import type { ImageDTO } from 'services/api/types';

export const createIsTooLargeToUpscaleSelector = (imageDTO?: ImageDTO) =>
  createMemoizedSelector(selectUpscalelice, selectConfigSlice, (upscale, config) => {
    const { upscaleModel, scale } = upscale;
    const { maxUpscalePixels } = config;

    if (!maxUpscalePixels || !upscaleModel || !imageDTO) {
      return false;
    }

    const upscaledPixels = imageDTO.width * scale * imageDTO.height * scale;
    return upscaledPixels > maxUpscalePixels;
  });

export const useIsTooLargeToUpscale = (imageDTO?: ImageDTO) => {
  const selectIsTooLargeToUpscale = useMemo(() => createIsTooLargeToUpscaleSelector(imageDTO), [imageDTO]);
  return useAppSelector(selectIsTooLargeToUpscale);
};
