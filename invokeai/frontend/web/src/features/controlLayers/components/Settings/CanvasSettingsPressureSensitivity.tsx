import { Checkbox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  selectPressureSensitivity,
  settingsPressureSensitivityToggled,
} from 'features/controlLayers/store/canvasSettingsSlice';
import type { ChangeEventHandler } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const CanvasSettingsPressureSensitivityCheckbox = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const pressureSensitivity = useAppSelector(selectPressureSensitivity);
  const onChange = useCallback<ChangeEventHandler<HTMLInputElement>>(() => {
    dispatch(settingsPressureSensitivityToggled());
  }, [dispatch]);

  return (
    <FormControl w="full">
      <FormLabel flexGrow={1}>{t('controlLayers.settings.pressureSensitivity')}</FormLabel>
      <Checkbox isChecked={pressureSensitivity} onChange={onChange} />
    </FormControl>
  );
});

CanvasSettingsPressureSensitivityCheckbox.displayName = 'CanvasSettingsPressureSensitivityCheckbox';
