import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import type { FillStyle } from 'features/controlLayers/store/types';
import { isFillStyle } from 'features/controlLayers/store/types';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

type Props = {
  style: FillStyle;
  onChange: (style: FillStyle) => void;
};

export const MaskFillStyle = memo(({ style, onChange }: Props) => {
  const { t } = useTranslation();
  const _onChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!isFillStyle(v?.value)) {
        return;
      }
      onChange(v.value);
    },
    [onChange]
  );

  const options = useMemo<ComboboxOption[]>(() => {
    return [
      {
        value: 'solid',
        label: t('controlLayers.fill.solid'),
      },
      {
        value: 'diagonal',
        label: t('controlLayers.fill.diagonal'),
      },
      {
        value: 'crosshatch',
        label: t('controlLayers.fill.crosshatch'),
      },
      {
        value: 'grid',
        label: t('controlLayers.fill.grid'),
      },
      {
        value: 'horizontal',
        label: t('controlLayers.fill.horizontal'),
      },
      {
        value: 'vertical',
        label: t('controlLayers.fill.vertical'),
      },
    ];
  }, [t]);

  const value = useMemo(() => options.find((o) => o.value === style), [options, style]);

  return (
    <FormControl>
      <FormLabel m={0}>{t('controlLayers.fill.fillStyle')}</FormLabel>
      <Combobox value={value} options={options} onChange={_onChange} isSearchable={false} />
    </FormControl>
  );
});

MaskFillStyle.displayName = 'MaskFillStyle';
