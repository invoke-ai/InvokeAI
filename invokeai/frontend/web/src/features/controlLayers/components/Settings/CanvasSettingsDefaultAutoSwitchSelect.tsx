import { FormControl, FormLabel, Select } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  selectDefaultAutoSwitch,
  settingsDefaultAutoSwitchChanged,
} from 'features/controlLayers/store/canvasSettingsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const CanvasSettingsDefaultAutoSwitchSelect = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const defaultAutoSwitch = useAppSelector(selectDefaultAutoSwitch);

  const onChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      const value = e.target.value as 'off' | 'switch_on_start' | 'switch_on_finish';
      dispatch(settingsDefaultAutoSwitchChanged(value));
    },
    [dispatch]
  );

  return (
    <FormControl>
      <FormLabel m={0}>{t('controlLayers.settings.defaultAutoSwitch')}</FormLabel>
      <Select size="sm" value={defaultAutoSwitch} onChange={onChange}>
        <option value="off">{t('controlLayers.autoSwitch.off')}</option>
        <option value="switch_on_start">{t('controlLayers.autoSwitch.switchOnStart')}</option>
        <option value="switch_on_finish">{t('controlLayers.autoSwitch.switchOnFinish')}</option>
      </Select>
    </FormControl>
  );
});

CanvasSettingsDefaultAutoSwitchSelect.displayName = 'CanvasSettingsDefaultAutoSwitchSelect';
