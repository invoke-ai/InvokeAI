import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { isRegionalGuidanceLayer, selectControlLayersSlice } from 'features/controlLayers/store/controlLayersSlice';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const selectValidLayerCount = createSelector(selectControlLayersSlice, (controlLayers) => {
  if (!controlLayers.present.isEnabled) {
    return 0;
  }
  const validLayers = controlLayers.present.layers
    .filter(isRegionalGuidanceLayer)
    .filter((l) => l.isEnabled)
    .filter((l) => {
      const hasTextPrompt = Boolean(l.positivePrompt || l.negativePrompt);
      const hasAtLeastOneImagePrompt = l.ipAdapters.length > 0;
      return hasTextPrompt || hasAtLeastOneImagePrompt;
    });

  return validLayers.length;
});

export const useControlLayersTitle = () => {
  const { t } = useTranslation();
  const validLayerCount = useAppSelector(selectValidLayerCount);
  const title = useMemo(() => {
    const suffix = validLayerCount > 0 ? ` (${validLayerCount})` : '';
    return `${t('controlLayers.controlLayers')}${suffix}`;
  }, [t, validLayerCount]);
  return title;
};
