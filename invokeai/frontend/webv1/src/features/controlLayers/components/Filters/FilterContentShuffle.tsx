import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import type { ContentShuffleFilterConfig } from 'features/controlLayers/store/filters';
import { IMAGE_FILTERS } from 'features/controlLayers/store/filters';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import type { FilterComponentProps } from './types';

type Props = FilterComponentProps<ContentShuffleFilterConfig>;
const DEFAULTS = IMAGE_FILTERS.content_shuffle.buildDefaults();

export const FilterContentShuffle = memo(({ onChange, config }: Props) => {
  const { t } = useTranslation();

  const handleScaleFactorChanged = useCallback(
    (v: number) => {
      onChange({ ...config, scale_factor: v });
    },
    [config, onChange]
  );

  return (
    <>
      <FormControl>
        <FormLabel m={0}>{t('controlLayers.filter.content_shuffle.scale_factor')}</FormLabel>
        <CompositeSlider
          value={config.scale_factor}
          defaultValue={DEFAULTS.scale_factor}
          onChange={handleScaleFactorChanged}
          min={0}
          max={4096}
          marks
        />
        <CompositeNumberInput
          value={config.scale_factor}
          defaultValue={DEFAULTS.scale_factor}
          onChange={handleScaleFactorChanged}
          min={0}
          max={4096}
        />
      </FormControl>
    </>
  );
});

FilterContentShuffle.displayName = 'FilterContentShuffle';
