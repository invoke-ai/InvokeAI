import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectGridSize, settingsGridSizeChanged } from 'features/controlLayers/store/canvasSettingsSlice';
import { isGridSize } from 'features/controlLayers/store/types';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const getValue = (valueString: string) => {
  switch (valueString) {
    case 'off':
      return 1;
    case '8':
      return 8;
    case '64':
      return 64;
    default:
      return null;
  }
};

export const CanvasSettingsGridSize = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const gridSize = useAppSelector(selectGridSize);
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
      if (!v) {
        return;
      }
      const value = getValue(v.value);
      if (!isGridSize(value)) {
        return;
      }
      dispatch(settingsGridSizeChanged(value));
    },
    [dispatch]
  );
  const value = useMemo(() => options.find((o) => getValue(o.value) === gridSize), [options, gridSize]);

  return (
    <FormControl>
      <FormLabel m={0} w="50%">
        {t('controlLayers.settings.snapToGrid.label')}
      </FormLabel>
      <Combobox options={options} value={value} onChange={onChange} isSearchable={false} />
    </FormControl>
  );
});

CanvasSettingsGridSize.displayName = 'CanvasSettingsSnapToGrid';
