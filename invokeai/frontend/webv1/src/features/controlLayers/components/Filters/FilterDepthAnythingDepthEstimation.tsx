import type { ComboboxOnChange } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import type { DepthAnythingFilterConfig, DepthAnythingModelSize } from 'features/controlLayers/store/filters';
import { isDepthAnythingModelSize } from 'features/controlLayers/store/filters';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import type { FilterComponentProps } from './types';

type Props = FilterComponentProps<DepthAnythingFilterConfig>;

export const FilterDepthAnythingDepthEstimation = memo(({ onChange, config }: Props) => {
  const { t } = useTranslation();
  const handleModelSizeChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!isDepthAnythingModelSize(v?.value)) {
        return;
      }
      onChange({ ...config, model_size: v.value });
    },
    [config, onChange]
  );

  const options: { label: string; value: DepthAnythingModelSize }[] = useMemo(
    () => [
      { label: t('controlLayers.filter.depth_anything_depth_estimation.model_size_small_v2'), value: 'small_v2' },
      { label: t('controlLayers.filter.depth_anything_depth_estimation.model_size_small'), value: 'small' },
      { label: t('controlLayers.filter.depth_anything_depth_estimation.model_size_base'), value: 'base' },
      { label: t('controlLayers.filter.depth_anything_depth_estimation.model_size_large'), value: 'large' },
    ],
    [t]
  );

  const value = useMemo(() => options.filter((o) => o.value === config.model_size)[0], [options, config.model_size]);

  return (
    <>
      <FormControl>
        <FormLabel m={0}>{t('controlLayers.filter.depth_anything_depth_estimation.model_size')}</FormLabel>
        <Combobox value={value} options={options} onChange={handleModelSizeChange} isSearchable={false} />
      </FormControl>
    </>
  );
});

FilterDepthAnythingDepthEstimation.displayName = 'FilterDepthAnythingDepthEstimation';
