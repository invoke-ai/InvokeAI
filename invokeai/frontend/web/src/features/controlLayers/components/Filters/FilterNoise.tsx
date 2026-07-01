import type { ComboboxOnChange } from '@invoke-ai/ui-library';
import { Combobox, CompositeNumberInput, CompositeSlider, FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import type { NoiseFilterConfig, NoiseTypes } from 'features/controlLayers/store/filters';
import { IMAGE_FILTERS, isNoiseTypes } from 'features/controlLayers/store/filters';
import type { ChangeEvent } from 'react';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import type { FilterComponentProps } from './types';

type Props = FilterComponentProps<NoiseFilterConfig>;
const DEFAULTS = IMAGE_FILTERS.img_noise.buildDefaults();

export const FilterNoise = memo(({ onChange, config }: Props) => {
  const { t } = useTranslation();
  const handleNoiseTypeChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!isNoiseTypes(v?.value)) {
        return;
      }
      onChange({ ...config, noise_type: v.value });
    },
    [config, onChange]
  );

  const handleAmountChange = useCallback(
    (v: number) => {
      onChange({ ...config, amount: v });
    },
    [config, onChange]
  );

  const handleColorChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      onChange({ ...config, noise_color: e.target.checked });
    },
    [config, onChange]
  );

  const handleSizeChange = useCallback(
    (v: number) => {
      onChange({ ...config, size: v });
    },
    [config, onChange]
  );

  const options: { label: string; value: NoiseTypes }[] = useMemo(
    () => [
      { label: t('controlLayers.filter.img_noise.gaussian_type'), value: 'gaussian' },
      { label: t('controlLayers.filter.img_noise.salt_and_pepper_type'), value: 'salt_and_pepper' },
    ],
    [t]
  );

  const value = useMemo(() => options.filter((o) => o.value === config.noise_type)[0], [options, config.noise_type]);

  return (
    <>
      <FormControl>
        <FormLabel m={0}>{t('controlLayers.filter.img_noise.noise_type')}</FormLabel>
        <Combobox value={value} options={options} onChange={handleNoiseTypeChange} isSearchable={false} />
      </FormControl>
      <FormControl>
        <FormLabel m={0}>{t('controlLayers.filter.img_noise.noise_amount')}</FormLabel>
        <CompositeSlider
          value={config.amount}
          defaultValue={DEFAULTS.amount}
          onChange={handleAmountChange}
          min={0}
          max={1}
          step={0.01}
          marks
        />
        <CompositeNumberInput
          value={config.amount}
          defaultValue={DEFAULTS.amount}
          onChange={handleAmountChange}
          min={0}
          max={1}
          step={0.01}
        />
      </FormControl>
      <FormControl>
        <FormLabel m={0}>{t('controlLayers.filter.img_noise.size')}</FormLabel>
        <CompositeSlider
          value={config.size}
          defaultValue={DEFAULTS.size}
          onChange={handleSizeChange}
          min={1}
          max={16}
          step={1}
          marks
        />
        <CompositeNumberInput
          value={config.size}
          defaultValue={DEFAULTS.size}
          onChange={handleSizeChange}
          min={1}
          max={256}
          step={1}
        />
      </FormControl>
      <FormControl w="max-content">
        <FormLabel m={0}>{t('controlLayers.filter.img_noise.noise_color')}</FormLabel>
        <Switch defaultChecked={DEFAULTS.noise_color} isChecked={config.noise_color} onChange={handleColorChange} />
      </FormControl>
    </>
  );
});

FilterNoise.displayName = 'Filternoise';
