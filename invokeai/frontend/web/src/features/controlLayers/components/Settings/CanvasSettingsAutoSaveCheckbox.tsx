import { Checkbox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectCanvasSettingsSlice, settingsAutoSaveToggled } from 'features/controlLayers/store/canvasSettingsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selectAutoSave = createSelector(selectCanvasSettingsSlice, (canvasSettings) => canvasSettings.autoSave);

export const CanvasSettingsAutoSaveCheckbox = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const autoSave = useAppSelector(selectAutoSave);
  const onChange = useCallback(() => dispatch(settingsAutoSaveToggled()), [dispatch]);
  return (
    <FormControl w="full">
      <FormLabel flexGrow={1}>{t('controlLayers.autoSave')}</FormLabel>
      <Checkbox isChecked={autoSave} onChange={onChange} />
    </FormControl>
  );
});

CanvasSettingsAutoSaveCheckbox.displayName = 'CanvasSettingsAutoSaveCheckbox';
