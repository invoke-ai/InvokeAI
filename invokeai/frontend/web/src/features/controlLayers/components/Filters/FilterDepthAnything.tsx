import type { ComboboxOnChange } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import type { DepthAnythingModelSize, DepthAnythingProcessorConfig } from 'features/controlLayers/store/types';
import { isDepthAnythingModelSize } from 'features/controlLayers/store/types';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import type { FilterComponentProps } from './types';

type Props = FilterComponentProps<DepthAnythingProcessorConfig>;

export const FilterDepthAnything = memo(({ onChange, config }: Props) => {
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
      { label: t('controlnet.depthAnythingSmallV2'), value: 'small_v2' },
      { label: t('controlnet.small'), value: 'small' },
      { label: t('controlnet.base'), value: 'base' },
      { label: t('controlnet.large'), value: 'large' },
    ],
    [t]
  );

  const value = useMemo(() => options.filter((o) => o.value === config.model_size)[0], [options, config.model_size]);

  return (
    <>
      <FormControl>
        <FormLabel m={0}>{t('controlnet.modelSize')}</FormLabel>
        <Combobox value={value} options={options} onChange={handleModelSizeChange} isSearchable={false} />
      </FormControl>
    </>
  );
});

FilterDepthAnything.displayName = 'FilterDepthAnything';
