import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { ImageDTO } from 'services/api/types';

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

const getMayUpscale = (
  upscaledPixels?: ReturnType<typeof getUpscaledPixels>,
  maxUpscalePixels?: number
) => {
  if (!upscaledPixels || !maxUpscalePixels) {
    return { x4: true, x2: true };
  }
  const mayUpscale = { x4: false, x2: false };
  if (upscaledPixels.x4 <= maxUpscalePixels) {
    mayUpscale.x4 = true;
  }
  if (upscaledPixels.x2 <= maxUpscalePixels) {
    mayUpscale.x2 = true;
  }

  return mayUpscale;
};

const getDetailTKey = (
  mayUpscale?: ReturnType<typeof getMayUpscale>,
  scaleFactor?: number
) => {
  if (!mayUpscale || !scaleFactor) {
    return;
  }

  if (mayUpscale.x4 && mayUpscale.x2) {
    return;
  }

  if (!mayUpscale.x2 && !mayUpscale.x4) {
    return 'parameters.mayUpscale.tooLarge';
  }

  if (!mayUpscale.x4 && mayUpscale.x2 && scaleFactor === 4) {
    return 'parameters.mayUpscale.useX2Model';
  }

  return;
};

export const createMayUpscaleSelector = (imageDTO?: ImageDTO) =>
  createSelector(
    stateSelector,
    ({ postprocessing, config }) => {
      const { esrganModelName } = postprocessing;
      const { maxUpscalePixels } = config;

      const upscaledPixels = getUpscaledPixels(imageDTO, maxUpscalePixels);
      const mayUpscale = getMayUpscale(upscaledPixels, maxUpscalePixels);
      const scaleFactor = esrganModelName.includes('x2') ? 2 : 4;
      const detailTKey = getDetailTKey(mayUpscale, scaleFactor);
      return {
        mayUpscale: scaleFactor === 2 ? mayUpscale.x2 : mayUpscale.x4,
        detailTKey,
      };
    },
    defaultSelectorOptions
  );

export const useMayUpscale = (imageDTO?: ImageDTO) => {
  const { t } = useTranslation();
  const selectMayUpscale = useMemo(
    () => createMayUpscaleSelector(imageDTO),
    [imageDTO]
  );
  const { mayUpscale, detailTKey } = useAppSelector(selectMayUpscale);

  return {
    mayUpscale,
    detail: detailTKey ? t(detailTKey) : undefined,
  };
};
