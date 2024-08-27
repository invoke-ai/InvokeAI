import { FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  selectCanvasSettingsSlice,
  settingsDynamicGridToggled,
} from 'features/controlLayers/store/canvasSettingsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selectDynamicGrid = createSelector(selectCanvasSettingsSlice, (canvasSettings) => canvasSettings.dynamicGrid);

export const CanvasSettingsDynamicGridSwitch = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const dynamicGrid = useAppSelector(selectDynamicGrid);
  const onChange = useCallback(() => {
    dispatch(settingsDynamicGridToggled());
  }, [dispatch]);

  return (
    <FormControl>
      <FormLabel m={0} flexGrow={1}>
        {t('controlLayers.dynamicGrid')}
      </FormLabel>
      <Switch size="sm" isChecked={dynamicGrid} onChange={onChange} />
    </FormControl>
  );
});

CanvasSettingsDynamicGridSwitch.displayName = 'CanvasSettingsDynamicGridSwitch';
