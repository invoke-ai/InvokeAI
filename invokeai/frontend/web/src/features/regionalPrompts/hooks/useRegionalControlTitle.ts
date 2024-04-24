import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { isVectorMaskLayer, selectRegionalPromptsSlice } from 'features/regionalPrompts/store/regionalPromptsSlice';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const selectValidLayerCount = createSelector(selectRegionalPromptsSlice, (regionalPrompts) => {
  if (!regionalPrompts.present.isEnabled) {
    return 0;
  }
  const validLayers = regionalPrompts.present.layers
    .filter(isVectorMaskLayer)
    .filter((l) => l.isVisible)
    .filter((l) => {
      const hasTextPrompt = Boolean(l.positivePrompt || l.negativePrompt);
      const hasAtLeastOneImagePrompt = l.ipAdapterIds.length > 0;
      return hasTextPrompt || hasAtLeastOneImagePrompt;
    });

  return validLayers.length;
});

export const useRegionalControlTitle = () => {
  const { t } = useTranslation();
  const validLayerCount = useAppSelector(selectValidLayerCount);
  const title = useMemo(() => {
    const suffix = validLayerCount > 0 ? ` (${validLayerCount})` : '';
    return `${t('regionalPrompts.regionalControl')}${suffix}`;
  }, [t, validLayerCount]);
  return title;
};
