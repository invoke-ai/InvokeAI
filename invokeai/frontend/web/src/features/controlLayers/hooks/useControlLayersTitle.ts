import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import {
  isControlAdapterLayer,
  isInitialImageLayer,
  isIPAdapterLayer,
  isRegionalGuidanceLayer,
  selectControlLayersSlice,
} from 'features/controlLayers/store/controlLayersSlice';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const selectValidLayerCount = createSelector(selectControlLayersSlice, (controlLayers) => {
  let count = 0;
  controlLayers.present.layers.forEach((l) => {
    if (isRegionalGuidanceLayer(l)) {
      const hasTextPrompt = Boolean(l.positivePrompt || l.negativePrompt);
      const hasAtLeastOneImagePrompt = l.ipAdapters.filter((ipa) => Boolean(ipa.image)).length > 0;
      if (hasTextPrompt || hasAtLeastOneImagePrompt) {
        count += 1;
      }
    }
    if (isControlAdapterLayer(l)) {
      if (l.controlAdapter.image || l.controlAdapter.processedImage) {
        count += 1;
      }
    }
    if (isIPAdapterLayer(l)) {
      if (l.ipAdapter.image) {
        count += 1;
      }
    }
    if (isInitialImageLayer(l)) {
      if (l.image) {
        count += 1;
      }
    }
  });

  return count;
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
