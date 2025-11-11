import { FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectShowHUD, settingsShowHUDToggled } from 'features/controlLayers/store/canvasSettingsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const CanvasSettingsShowHUDSwitch = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const showHUD = useAppSelector((state) => selectShowHUD(state));
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
