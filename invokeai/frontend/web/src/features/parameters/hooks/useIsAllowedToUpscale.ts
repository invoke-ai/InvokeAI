import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { selectPostprocessingSlice } from 'features/parameters/store/postprocessingSlice';
import { selectConfigSlice } from 'features/system/store/configSlice';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import type { ImageDTO } from 'services/api/types';

const getUpscaledPixels = (imageDTO?: ImageDTO, maxUpscalePixels?: number) => {
  if (!imageDTO) {
    return;
  }
  if (!maxUpscalePixels) {
    return;
  }
  const { width, height } = imageDTO;
  const x4 = height * 4 * width * 4;
  const x2 = height * 2 * width * 2;
  return { x4, x2 };
};

const getIsAllowedToUpscale = (upscaledPixels?: ReturnType<typeof getUpscaledPixels>, maxUpscalePixels?: number) => {
  if (!upscaledPixels || !maxUpscalePixels) {
    return { x4: true, x2: true };
  }
  const isAllowedToUpscale = { x4: false, x2: false };
  if (upscaledPixels.x4 <= maxUpscalePixels) {
    isAllowedToUpscale.x4 = true;
  }
  if (upscaledPixels.x2 <= maxUpscalePixels) {
    isAllowedToUpscale.x2 = true;
  }

  return isAllowedToUpscale;
};

const getDetailTKey = (isAllowedToUpscale?: ReturnType<typeof getIsAllowedToUpscale>, scaleFactor?: number) => {
  if (!isAllowedToUpscale || !scaleFactor) {
    return;
  }

  if (isAllowedToUpscale.x4 && isAllowedToUpscale.x2) {
    return;
  }

  if (!isAllowedToUpscale.x2 && !isAllowedToUpscale.x4) {
    return 'parameters.isAllowedToUpscale.tooLarge';
  }

  if (!isAllowedToUpscale.x4 && isAllowedToUpscale.x2 && scaleFactor === 4) {
    return 'parameters.isAllowedToUpscale.useX2Model';
  }

  return;
};

export const createIsAllowedToUpscaleSelector = (imageDTO?: ImageDTO) =>
  createMemoizedSelector(selectPostprocessingSlice, selectConfigSlice, (postprocessing, config) => {
    const { esrganModelName } = postprocessing;
    const { maxUpscalePixels } = config;

    const upscaledPixels = getUpscaledPixels(imageDTO, maxUpscalePixels);
    const isAllowedToUpscale = getIsAllowedToUpscale(upscaledPixels, maxUpscalePixels);
    const scaleFactor = esrganModelName.includes('x2') ? 2 : 4;
    const detailTKey = getDetailTKey(isAllowedToUpscale, scaleFactor);
    return {
      isAllowedToUpscale: scaleFactor === 2 ? isAllowedToUpscale.x2 : isAllowedToUpscale.x4,
      detailTKey,
    };
  });

export const useIsAllowedToUpscale = (imageDTO?: ImageDTO) => {
  const { t } = useTranslation();
  const selectIsAllowedToUpscale = useMemo(() => createIsAllowedToUpscaleSelector(imageDTO), [imageDTO]);
  const { isAllowedToUpscale, detailTKey } = useAppSelector(selectIsAllowedToUpscale);

  return {
    isAllowedToUpscale,
    detail: detailTKey ? t(detailTKey) : undefined,
  };
};
