import { FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectCanvasSettingsSlice, settingsShowHUDToggled } from 'features/controlLayers/store/canvasSettingsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selectShowHUD = createSelector(selectCanvasSettingsSlice, (canvasSettings) => canvasSettings.showHUD);

export const CanvasSettingsShowHUDSwitch = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const showHUD = useAppSelector(selectShowHUD);
  const onChange = useCallback(() => {
    dispatch(settingsShowHUDToggled());
  }, [dispatch]);

  return (
    <FormControl>
      <FormLabel m={0} flexGrow={1}>
        {t('controlLayers.showHUD')}
      </FormLabel>
      <Switch size="sm" isChecked={showHUD} onChange={onChange} />
    </FormControl>
  );
});

CanvasSettingsShowHUDSwitch.displayName = 'CanvasSettingsShowHUDSwitch';
