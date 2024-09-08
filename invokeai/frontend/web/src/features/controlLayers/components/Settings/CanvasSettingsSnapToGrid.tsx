import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectSnapToGrid, settingsSnapToGridChanged } from 'features/controlLayers/store/canvasSettingsSlice';
import { isSnap } from 'features/controlLayers/store/types';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export const CanvasSettingsSnapToGrid = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const snapToGrid = useAppSelector(selectSnapToGrid);
  const options = useMemo<ComboboxOption[]>(
    () => [
      { label: t('controlLayers.settings.snapToGrid.off'), value: 'off' },
      { label: t('controlLayers.settings.snapToGrid.8'), value: '8' },
      { label: t('controlLayers.settings.snapToGrid.64'), value: '64' },
    ],
    [t]
  );
  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!isSnap(v?.value)) {
        return;
      }
      dispatch(settingsSnapToGridChanged(v.value));
    },
    [dispatch]
  );
  const value = useMemo(() => options.find((o) => o.value === snapToGrid), [options, snapToGrid]);

  return (
    <FormControl>
      <FormLabel m={0} w="50%">
        {t('controlLayers.settings.snapToGrid.label')}
      </FormLabel>
      <Combobox options={options} value={value} onChange={onChange} isSearchable={false} />
    </FormControl>
  );
});

CanvasSettingsSnapToGrid.displayName = 'CanvasSettingsSnapToGrid';
