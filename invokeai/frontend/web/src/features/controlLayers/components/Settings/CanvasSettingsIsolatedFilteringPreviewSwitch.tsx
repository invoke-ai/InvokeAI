import { FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  selectIsolatedFilteringPreview,
  settingsIsolatedFilteringPreviewToggled,
} from 'features/controlLayers/store/canvasSettingsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const CanvasSettingsIsolatedFilteringPreviewSwitch = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isolatedFilteringPreview = useAppSelector(selectIsolatedFilteringPreview);
  const onChange = useCallback(() => {
    dispatch(settingsIsolatedFilteringPreviewToggled());
  }, [dispatch]);

  return (
    <FormControl>
      <FormLabel m={0} flexGrow={1}>
        {t('controlLayers.settings.isolatedFilteringPreview')}
      </FormLabel>
      <Switch size="sm" isChecked={isolatedFilteringPreview} onChange={onChange} />
    </FormControl>
  );
});

CanvasSettingsIsolatedFilteringPreviewSwitch.displayName = 'CanvasSettingsIsolatedFilteringPreviewSwitch';
