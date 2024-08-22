import { FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { settingsDynamicGridToggled } from 'features/controlLayers/store/canvasV2Slice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const CanvasSettingsDynamicGridSwitch = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const dynamicGrid = useAppSelector((s) => s.canvasV2.settings.dynamicGrid);
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
