import type { ComboboxOnChange } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import type { IPMethodV2 } from 'features/controlLayers/store/types';
import { isIPMethodV2 } from 'features/controlLayers/store/types';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { assert } from 'tsafe';

type Props = {
  method: IPMethodV2;
  onChange: (method: IPMethodV2) => void;
};

export const IPAdapterMethod = memo(({ method, onChange }: Props) => {
  const { t } = useTranslation();
  const options: { label: string; value: IPMethodV2 }[] = useMemo(
    () => [
      { label: t('controlLayers.ipAdapterMethod.full'), value: 'full' },
      { label: t('controlLayers.ipAdapterMethod.style'), value: 'style' },
      { label: t('controlLayers.ipAdapterMethod.composition'), value: 'composition' },
    ],
    [t]
  );
  const _onChange = useCallback<ComboboxOnChange>(
    (v) => {
      assert(isIPMethodV2(v?.value));
      onChange(v.value);
    },
    [onChange]
  );
  const value = useMemo(() => options.find((o) => o.value === method), [options, method]);

  return (
    <FormControl>
      <InformationalPopover feature="ipAdapterMethod">
        <FormLabel>{t('controlLayers.ipAdapterMethod.ipAdapterMethod')}</FormLabel>
      </InformationalPopover>
      <Combobox value={value} options={options} onChange={_onChange} />
    </FormControl>
  );
});

IPAdapterMethod.displayName = 'IPAdapterMethod';
