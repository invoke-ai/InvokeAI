import { FormControl, FormLabel, Switch, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  selectIsolatedLayerPreview,
  settingsIsolatedLayerPreviewToggled,
} from 'features/controlLayers/store/canvasSettingsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const CanvasOperationIsolatedLayerPreviewSwitch = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isolatedLayerPreview = useAppSelector(selectIsolatedLayerPreview);
  const onChangeIsolatedPreview = useCallback(() => {
    dispatch(settingsIsolatedLayerPreviewToggled());
  }, [dispatch]);

  return (
    <Tooltip label={t('controlLayers.settings.isolatedLayerPreviewDesc')}>
      <FormControl w="min-content">
        <FormLabel m={0}>{t('controlLayers.settings.isolatedPreview')}</FormLabel>
        <Switch size="sm" isChecked={isolatedLayerPreview} onChange={onChangeIsolatedPreview} />
      </FormControl>
    </Tooltip>
  );
});

CanvasOperationIsolatedLayerPreviewSwitch.displayName = 'CanvasOperationIsolatedLayerPreviewSwitch';
