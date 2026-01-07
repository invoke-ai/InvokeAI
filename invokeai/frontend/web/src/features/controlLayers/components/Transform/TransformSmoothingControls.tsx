import { Flex, FormControl, FormLabel, Select, Switch, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  selectTransformSmoothingEnabled,
  selectTransformSmoothingMode,
  settingsTransformSmoothingEnabledToggled,
  settingsTransformSmoothingModeChanged,
  type TransformSmoothingMode,
} from 'features/controlLayers/store/canvasSettingsSlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const TransformSmoothingControls = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const smoothingEnabled = useAppSelector(selectTransformSmoothingEnabled);
  const smoothingMode = useAppSelector(selectTransformSmoothingMode);

  const onToggle = useCallback(() => {
    dispatch(settingsTransformSmoothingEnabledToggled());
  }, [dispatch]);

  const onModeChange = useCallback(
    (e: ChangeEvent<HTMLSelectElement>) => {
      dispatch(settingsTransformSmoothingModeChanged(e.target.value as TransformSmoothingMode));
    },
    [dispatch]
  );

  return (
    <Flex w="full" gap={4} alignItems="center" flexWrap="wrap">
      <Tooltip label={t('controlLayers.transform.smoothingDesc')}>
        <FormControl w="min-content">
          <FormLabel m={0}>{t('controlLayers.transform.smoothing')}</FormLabel>
          <Switch size="sm" isChecked={smoothingEnabled} onChange={onToggle} />
        </FormControl>
      </Tooltip>
      <FormControl flex={1} minW={200} maxW={280}>
        <FormLabel m={0}>{t('controlLayers.transform.smoothingMode')}</FormLabel>
        <Select size="sm" value={smoothingMode} onChange={onModeChange} isDisabled={!smoothingEnabled}>
          <option value="nearest">{t('controlLayers.transform.smoothingModeNearest')}</option>
          <option value="bilinear">{t('controlLayers.transform.smoothingModeBilinear')}</option>
          <option value="bicubic">{t('controlLayers.transform.smoothingModeBicubic')}</option>
          <option value="lanczos">{t('controlLayers.transform.smoothingModeLanczos')}</option>
        </Select>
      </FormControl>
    </Flex>
  );
});

TransformSmoothingControls.displayName = 'TransformSmoothingControls';
