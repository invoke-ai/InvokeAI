import { FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  selectIsolatedTransformingPreview,
  settingsIsolatedTransformingPreviewToggled,
} from 'features/controlLayers/store/canvasSettingsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const CanvasSettingsIsolatedTransformingPreviewSwitch = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isolatedTransformingPreview = useAppSelector(selectIsolatedTransformingPreview);
  const onChange = useCallback(() => {
    dispatch(settingsIsolatedTransformingPreviewToggled());
  }, [dispatch]);

  return (
    <FormControl>
      <FormLabel m={0} flexGrow={1}>
        {t('controlLayers.settings.isolatedTransformingPreview')}
      </FormLabel>
      <Switch size="sm" isChecked={isolatedTransformingPreview} onChange={onChange} />
    </FormControl>
  );
});

CanvasSettingsIsolatedTransformingPreviewSwitch.displayName = 'CanvasSettingsIsolatedTransformingPreviewSwitch';
