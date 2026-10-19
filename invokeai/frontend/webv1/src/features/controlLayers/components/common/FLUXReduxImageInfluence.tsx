import type { ComboboxOnChange } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import type { FLUXReduxImageInfluence as FLUXReduxImageInfluenceType } from 'features/controlLayers/store/types';
import { isFLUXReduxImageInfluence } from 'features/controlLayers/store/types';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { assert } from 'tsafe';

type Props = {
  imageInfluence: FLUXReduxImageInfluenceType;
  onChange: (imageInfluence: FLUXReduxImageInfluenceType) => void;
};

export const FLUXReduxImageInfluence = memo(({ imageInfluence, onChange }: Props) => {
  const { t } = useTranslation();

  const options = useMemo(
    () =>
      [
        {
          label: t('controlLayers.fluxReduxImageInfluence.lowest'),
          value: 'lowest',
        },
        {
          label: t('controlLayers.fluxReduxImageInfluence.low'),
          value: 'low',
        },
        {
          label: t('controlLayers.fluxReduxImageInfluence.medium'),
          value: 'medium',
        },
        {
          label: t('controlLayers.fluxReduxImageInfluence.high'),
          value: 'high',
        },
        {
          label: t('controlLayers.fluxReduxImageInfluence.highest'),
          value: 'highest',
        },
      ] satisfies { label: string; value: FLUXReduxImageInfluenceType }[],
    [t]
  );
  const _onChange = useCallback<ComboboxOnChange>(
    (v) => {
      assert(isFLUXReduxImageInfluence(v?.value));
      onChange(v.value);
    },
    [onChange]
  );
  const value = useMemo(() => options.find((o) => o.value === imageInfluence), [options, imageInfluence]);

  return (
    <FormControl>
      <FormLabel m={0}>{t('controlLayers.fluxReduxImageInfluence.imageInfluence')}</FormLabel>
      <Combobox value={value} options={options} onChange={_onChange} />
    </FormControl>
  );
});

FLUXReduxImageInfluence.displayName = 'FLUXReduxImageInfluence';
