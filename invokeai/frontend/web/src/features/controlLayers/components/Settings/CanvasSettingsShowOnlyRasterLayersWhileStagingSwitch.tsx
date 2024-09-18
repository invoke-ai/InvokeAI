import { FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  selectShowOnlyRasterLayersWhileStaging,
  settingsShowOnlyRasterLayersWhileStagingToggled,
} from 'features/controlLayers/store/canvasSettingsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const CanvasSettingsShowOnlyRasterLayersWhileStagingSwitch = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const showOnlyRasterLayersWhileStaging = useAppSelector(selectShowOnlyRasterLayersWhileStaging);
  const onChange = useCallback(() => {
    dispatch(settingsShowOnlyRasterLayersWhileStagingToggled());
  }, [dispatch]);

  return (
    <FormControl>
      <FormLabel m={0} flexGrow={1}>
        {t('controlLayers.settings.showOnlyRasterLayersWhileStaging')}
      </FormLabel>
      <Switch size="sm" isChecked={showOnlyRasterLayersWhileStaging} onChange={onChange} />
    </FormControl>
  );
});

CanvasSettingsShowOnlyRasterLayersWhileStagingSwitch.displayName =
  'CanvasSettingsShowOnlyRasterLayersWhileStagingSwitch';
