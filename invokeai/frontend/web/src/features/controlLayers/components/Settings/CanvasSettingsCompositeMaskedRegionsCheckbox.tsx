import { Checkbox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  selectCanvasSettingsSlice,
  settingsCompositeMaskedRegionsChanged,
} from 'features/controlLayers/store/canvasSettingsSlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selectCompositeMaskedRegions = createSelector(
  selectCanvasSettingsSlice,
  (canvasSettings) => canvasSettings.compositeMaskedRegions
);

export const CanvasSettingsCompositeMaskedRegionsCheckbox = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const compositeMaskedRegions = useAppSelector(selectCompositeMaskedRegions);
  const onChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => dispatch(settingsCompositeMaskedRegionsChanged(e.target.checked)),
    [dispatch]
  );
  return (
    <FormControl w="full">
      <FormLabel flexGrow={1}>{t('controlLayers.compositeMaskedRegions')}</FormLabel>
      <Checkbox isChecked={compositeMaskedRegions} onChange={onChange} />
    </FormControl>
  );
});

CanvasSettingsCompositeMaskedRegionsCheckbox.displayName = 'CanvasSettingsCompositeMaskedRegionsCheckbox';
