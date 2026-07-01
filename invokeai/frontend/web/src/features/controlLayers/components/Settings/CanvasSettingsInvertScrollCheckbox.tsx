import { Checkbox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  selectCanvasSettingsSlice,
  settingsInvertScrollForToolWidthChanged,
} from 'features/controlLayers/store/canvasSettingsSlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selectInvertScrollForToolWidth = createSelector(
  selectCanvasSettingsSlice,
  (settings) => settings.invertScrollForToolWidth
);

export const CanvasSettingsInvertScrollCheckbox = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const invertScrollForToolWidth = useAppSelector(selectInvertScrollForToolWidth);
  const onChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(settingsInvertScrollForToolWidthChanged(e.target.checked));
    },
    [dispatch]
  );
  return (
    <FormControl w="full">
      <FormLabel flexGrow={1}>{t('controlLayers.settings.invertBrushSizeScrollDirection')}</FormLabel>
      <Checkbox isChecked={invertScrollForToolWidth} onChange={onChange} />
    </FormControl>
  );
});

CanvasSettingsInvertScrollCheckbox.displayName = 'CanvasSettingsInvertScrollCheckbox';
