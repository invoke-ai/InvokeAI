import type { ComboboxOnChange } from '@invoke-ai/ui-library';
import { Combobox, CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import type { BlurFilterConfig, BlurTypes } from 'features/controlLayers/store/filters';
import { IMAGE_FILTERS, isBlurTypes } from 'features/controlLayers/store/filters';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import type { FilterComponentProps } from './types';

type Props = FilterComponentProps<BlurFilterConfig>;
const DEFAULTS = IMAGE_FILTERS.img_blur.buildDefaults();

export const FilterBlur = memo(({ onChange, config }: Props) => {
  const { t } = useTranslation();
  const handleBlurTypeChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!isBlurTypes(v?.value)) {
        return;
      }
      onChange({ ...config, blur_type: v.value });
    },
    [config, onChange]
  );

  const handleRadiusChange = useCallback(
    (v: number) => {
      onChange({ ...config, radius: v });
    },
    [config, onChange]
  );

  const options: { label: string; value: BlurTypes }[] = useMemo(
    () => [
      { label: t('controlLayers.filter.img_blur.gaussian_type'), value: 'gaussian' },
      { label: t('controlLayers.filter.img_blur.box_type'), value: 'box' },
    ],
    [t]
  );

  const value = useMemo(() => options.filter((o) => o.value === config.blur_type)[0], [options, config.blur_type]);

  return (
    <>
      <FormControl>
        <FormLabel m={0}>{t('controlLayers.filter.img_blur.blur_type')}</FormLabel>
        <Combobox value={value} options={options} onChange={handleBlurTypeChange} isSearchable={false} />
      </FormControl>
      <FormControl>
        <FormLabel m={0}>{t('controlLayers.filter.img_blur.blur_radius')}</FormLabel>
        <CompositeSlider
          value={config.radius}
          defaultValue={DEFAULTS.radius}
          onChange={handleRadiusChange}
          min={1}
          max={64}
          step={0.1}
          marks
        />
        <CompositeNumberInput
          value={config.radius}
          defaultValue={DEFAULTS.radius}
          onChange={handleRadiusChange}
          min={1}
          max={4096}
          step={0.1}
        />
      </FormControl>
    </>
  );
});

FilterBlur.displayName = 'FilterBlur';
