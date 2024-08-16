import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { canvasBackgroundStyleChanged } from 'features/controlLayers/store/canvasV2Slice';
import { isCanvasBackgroundStyle } from 'features/controlLayers/store/types';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export const CanvasSettingsBackgroundStyle = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const canvasBackgroundStyle = useAppSelector((s) => s.canvasV2.settings.canvasBackgroundStyle);
  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!isCanvasBackgroundStyle(v?.value)) {
        return;
      }
      dispatch(canvasBackgroundStyleChanged(v.value));
    },
    [dispatch]
  );

  const options = useMemo<ComboboxOption[]>(() => {
    return [
      {
        value: 'solid',
        label: t('controlLayers.background.solid'),
      },
      {
        value: 'checkerboard',
        label: t('controlLayers.background.checkerboard'),
      },
      {
        value: 'dynamicGrid',
        label: t('controlLayers.background.dynamicGrid'),
      },
    ];
  }, [t]);

  const value = useMemo(() => options.find((o) => o.value === canvasBackgroundStyle), [options, canvasBackgroundStyle]);

  return (
    <FormControl orientation="vertical">
      <FormLabel m={0}>{t('controlLayers.background.backgroundStyle')}</FormLabel>
      <Combobox value={value} options={options} onChange={onChange} isSearchable={false} />
    </FormControl>
  );
});

CanvasSettingsBackgroundStyle.displayName = 'CanvasSettingsBackgroundStyle';
