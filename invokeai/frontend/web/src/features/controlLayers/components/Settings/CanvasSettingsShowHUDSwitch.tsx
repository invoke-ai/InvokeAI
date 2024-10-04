import { FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  selectCanvasSettingsSlice,
  settingsShowHUDToggled,
  settingsShowSystemStatsToggled,
} from 'features/controlLayers/store/canvasSettingsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selectShowHUD = createSelector(selectCanvasSettingsSlice, (canvasSettings) => canvasSettings.showHUD);
const selectShowSystemStats = createSelector(
  selectCanvasSettingsSlice,
  (canvasSettings) => canvasSettings.showSystemStats
);

export const CanvasSettingsShowHUDSwitch = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const showHUD = useAppSelector(selectShowHUD);
  const showSystemStats = useAppSelector(selectShowSystemStats);

  const onToggleHUD = useCallback(() => {
    dispatch(settingsShowHUDToggled());
  }, [dispatch]);

  const onToggleSystemStats = useCallback(() => {
    dispatch(settingsShowSystemStatsToggled());
  }, [dispatch]);

  return (
    <div>
      <FormControl>
        <FormLabel m={0} flexGrow={1}>
          {t('controlLayers.showHUD')}
        </FormLabel>
        <Switch size="sm" isChecked={showHUD} onChange={onToggleHUD} />
      </FormControl>

      {/* Show the System Stats toggle only if Show HUD is enabled */}
      {showHUD && (
        <FormControl mt={2}>
          <FormLabel m={0} flexGrow={1}>
            {t('controlLayers.showSystemStats')}
          </FormLabel>
          <Switch size="sm" isChecked={showSystemStats} onChange={onToggleSystemStats} />
        </FormControl>
      )}
    </div>
  );
});

CanvasSettingsShowHUDSwitch.displayName = 'CanvasSettingsShowHUDSwitch';
