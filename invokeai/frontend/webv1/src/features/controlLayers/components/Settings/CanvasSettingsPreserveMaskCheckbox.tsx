import { Checkbox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectPreserveMask, settingsPreserveMaskToggled } from 'features/controlLayers/store/canvasSettingsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const CanvasSettingsPreserveMaskCheckbox = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const preserveMask = useAppSelector(selectPreserveMask);
  const onChange = useCallback(() => dispatch(settingsPreserveMaskToggled()), [dispatch]);
  return (
    <FormControl w="full">
      <FormLabel flexGrow={1}>{t('controlLayers.settings.preserveMask.label')}</FormLabel>
      <Checkbox isChecked={preserveMask} onChange={onChange} />
    </FormControl>
  );
});

CanvasSettingsPreserveMaskCheckbox.displayName = 'CanvasSettingsPreserveMaskCheckbox';
