import type { ComboboxOnChange } from '@invoke-ai/ui-library';
import { Combobox, CompositeNumberInput, CompositeSlider, FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import type { AdjustImageFilterConfig, AjustImageChannels } from 'features/controlLayers/store/filters';
import { IMAGE_FILTERS, isAjustImageChannels } from 'features/controlLayers/store/filters';
import type { ChangeEvent } from 'react';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import type { FilterComponentProps } from './types';

type Props = FilterComponentProps<AdjustImageFilterConfig>;
const DEFAULTS = IMAGE_FILTERS.adjust_image.buildDefaults();

export const FilterAdjustImage = memo(({ onChange, config }: Props) => {
  const { t } = useTranslation();
  const handleChannelChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!isAjustImageChannels(v?.value)) {
        return;
      }
      onChange({ ...config, channel: v.value });
    },
    [config, onChange]
  );

  const handleValueChange = useCallback(
    (v: number) => {
      onChange({ ...config, value: v });
    },
    [config, onChange]
  );

  const handleScaleChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      onChange({ ...config, scale_values: e.target.checked });
    },
    [config, onChange]
  );

  const options: { label: string; value: AjustImageChannels }[] = useMemo(
    () => [
      { label: t('controlLayers.filter.adjust_image.red'), value: 'Red (RGBA)' },
      { label: t('controlLayers.filter.adjust_image.green'), value: 'Green (RGBA)' },
      { label: t('controlLayers.filter.adjust_image.blue'), value: 'Blue (RGBA)' },
      { label: t('controlLayers.filter.adjust_image.alpha'), value: 'Alpha (RGBA)' },
      { label: t('controlLayers.filter.adjust_image.cyan'), value: 'Cyan (CMYK)' },
      { label: t('controlLayers.filter.adjust_image.magenta'), value: 'Magenta (CMYK)' },
      { label: t('controlLayers.filter.adjust_image.yellow'), value: 'Yellow (CMYK)' },
      { label: t('controlLayers.filter.adjust_image.black'), value: 'Black (CMYK)' },
      { label: t('controlLayers.filter.adjust_image.hue'), value: 'Hue (HSV)' },
      { label: t('controlLayers.filter.adjust_image.saturation'), value: 'Saturation (HSV)' },
      { label: t('controlLayers.filter.adjust_image.value'), value: 'Value (HSV)' },
      { label: t('controlLayers.filter.adjust_image.luminosity'), value: 'Luminosity (LAB)' },
      { label: t('controlLayers.filter.adjust_image.a'), value: 'A (LAB)' },
      { label: t('controlLayers.filter.adjust_image.b'), value: 'B (LAB)' },
      { label: t('controlLayers.filter.adjust_image.y'), value: 'Y (YCbCr)' },
      { label: t('controlLayers.filter.adjust_image.cb'), value: 'Cb (YCbCr)' },
      { label: t('controlLayers.filter.adjust_image.cr'), value: 'Cr (YCbCr)' },
    ],
    [t]
  );

  const value = useMemo(() => options.filter((o) => o.value === config.channel)[0], [options, config.channel]);

  return (
    <>
      <FormControl>
        <FormLabel m={0}>{t('controlLayers.filter.adjust_image.channel')}</FormLabel>
        <Combobox value={value} options={options} onChange={handleChannelChange} isSearchable={false} />
      </FormControl>
      <FormControl>
        <FormLabel m={0}>{t('controlLayers.filter.adjust_image.value_setting')}</FormLabel>
        <CompositeSlider
          value={config.value}
          defaultValue={DEFAULTS.value}
          onChange={handleValueChange}
          min={0}
          max={2}
          step={0.0025}
          marks
        />
        <CompositeNumberInput
          value={config.value}
          defaultValue={DEFAULTS.value}
          onChange={handleValueChange}
          min={0}
          max={255}
          step={0.0025}
        />
      </FormControl>
      <FormControl w="max-content">
        <FormLabel m={0}>{t('controlLayers.filter.adjust_image.scale_values')}</FormLabel>
        <Switch defaultChecked={DEFAULTS.scale_values} isChecked={config.scale_values} onChange={handleScaleChange} />
      </FormControl>
    </>
  );
});

FilterAdjustImage.displayName = 'FilterAdjustImage';
