import { FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  selectIsolatedStagingPreview,
  settingsIsolatedStagingPreviewToggled,
} from 'features/controlLayers/store/canvasSettingsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const CanvasSettingsIsolatedStagingPreviewSwitch = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isolatedStagingPreview = useAppSelector(selectIsolatedStagingPreview);
  const onChange = useCallback(() => {
    dispatch(settingsIsolatedStagingPreviewToggled());
  }, [dispatch]);

  return (
    <FormControl>
      <FormLabel m={0} flexGrow={1}>
        {t('controlLayers.settings.isolatedStagingPreview')}
      </FormLabel>
      <Switch size="sm" isChecked={isolatedStagingPreview} onChange={onChange} />
    </FormControl>
  );
});

CanvasSettingsIsolatedStagingPreviewSwitch.displayName = 'CanvasSettingsIsolatedStagingPreviewSwitch';
