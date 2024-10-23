import { FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  selectIsolatedLayerPreview,
  settingsIsolatedLayerPreviewToggled,
} from 'features/controlLayers/store/canvasSettingsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const CanvasSettingsIsolatedLayerPreviewSwitch = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isolatedLayerPreview = useAppSelector(selectIsolatedLayerPreview);
  const onChange = useCallback(() => {
    dispatch(settingsIsolatedLayerPreviewToggled());
  }, [dispatch]);

  return (
    <FormControl>
      <FormLabel m={0} flexGrow={1}>
        {t('controlLayers.settings.isolatedLayerPreview')}
      </FormLabel>
      <Switch size="sm" isChecked={isolatedLayerPreview} onChange={onChange} />
    </FormControl>
  );
});

CanvasSettingsIsolatedLayerPreviewSwitch.displayName = 'CanvasSettingsIsolatedLayerPreviewSwitch';
