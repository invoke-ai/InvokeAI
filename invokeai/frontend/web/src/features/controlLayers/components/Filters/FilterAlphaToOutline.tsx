import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import type { AlphaToOutlineFilterConfig } from 'features/controlLayers/store/filters';
import { IMAGE_FILTERS } from 'features/controlLayers/store/filters';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import type { FilterComponentProps } from './types';

type Props = FilterComponentProps<AlphaToOutlineFilterConfig>;
const DEFAULTS = IMAGE_FILTERS.alpha_to_outline.buildDefaults();

export const FilterAlphaToOutline = ({ onChange, config }: Props) => {
  const { t } = useTranslation();

  const handleInnerLineWidthPercentChange = useCallback(
    (v: number) => {
      onChange({ ...config, inner_line_width_percent: v });
    },
    [onChange, config]
  );

  const handleOuterLineWidthPercentChange = useCallback(
    (v: number) => {
      onChange({ ...config, outer_line_width_percent: v });
    },
    [onChange, config]
  );

  return (
    <>
      <FormControl>
        <FormLabel m={0}>{t('controlLayers.filter.alpha_to_outline.inner_line_width_percent')}</FormLabel>
        <CompositeSlider
          value={config.inner_line_width_percent}
          onChange={handleInnerLineWidthPercentChange}
          defaultValue={DEFAULTS.inner_line_width_percent}
          min={0}
          max={100}
        />
        <CompositeNumberInput
          value={config.inner_line_width_percent}
          onChange={handleInnerLineWidthPercentChange}
          defaultValue={DEFAULTS.inner_line_width_percent}
          min={0}
          max={100}
        />
      </FormControl>
      <FormControl>
        <FormLabel m={0}>{t('controlLayers.filter.alpha_to_outline.outer_line_width_percent')}</FormLabel>
        <CompositeSlider
          value={config.outer_line_width_percent}
          onChange={handleOuterLineWidthPercentChange}
          defaultValue={DEFAULTS.outer_line_width_percent}
          min={0}
          max={100}
        />
        <CompositeNumberInput
          value={config.outer_line_width_percent}
          onChange={handleOuterLineWidthPercentChange}
          defaultValue={DEFAULTS.outer_line_width_percent}
          min={0}
          max={100}
        />
      </FormControl>
    </>
  );
};

FilterAlphaToOutline.displayName = 'FilterAlphaToOutline';
