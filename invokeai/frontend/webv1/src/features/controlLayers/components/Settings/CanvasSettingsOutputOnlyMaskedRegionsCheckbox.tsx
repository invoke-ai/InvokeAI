import { Checkbox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  selectOutputOnlyMaskedRegions,
  settingsOutputOnlyMaskedRegionsToggled,
} from 'features/controlLayers/store/canvasSettingsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const CanvasSettingsOutputOnlyMaskedRegionsCheckbox = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const outputOnlyMaskedRegions = useAppSelector(selectOutputOnlyMaskedRegions);
  const onChange = useCallback(() => {
    dispatch(settingsOutputOnlyMaskedRegionsToggled());
  }, [dispatch]);
  return (
    <FormControl w="full">
      <FormLabel flexGrow={1}>{t('controlLayers.outputOnlyMaskedRegions')}</FormLabel>
      <Checkbox isChecked={outputOnlyMaskedRegions} onChange={onChange} />
    </FormControl>
  );
});

CanvasSettingsOutputOnlyMaskedRegionsCheckbox.displayName = 'CanvasSettingsOutputOnlyMaskedRegionsCheckbox';
