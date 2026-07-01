import { Checkbox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectSnapToGrid, settingsSnapToGridToggled } from 'features/controlLayers/store/canvasSettingsSlice';
import type { ChangeEventHandler } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const CanvasSettingsSnapToGridCheckbox = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const snapToGrid = useAppSelector(selectSnapToGrid);
  const onChange = useCallback<ChangeEventHandler<HTMLInputElement>>(() => {
    dispatch(settingsSnapToGridToggled());
  }, [dispatch]);

  return (
    <FormControl w="full">
      <FormLabel flexGrow={1}>{t('controlLayers.settings.snapToGrid.label')}</FormLabel>
      <Checkbox isChecked={snapToGrid} onChange={onChange} />
    </FormControl>
  );
});

CanvasSettingsSnapToGridCheckbox.displayName = 'CanvasSettingsSnapToGrid';
