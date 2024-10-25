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
  const handleLineWidthChange = useCallback(
    (v: number) => {
      onChange({ ...config, line_width: v });
    },
    [onChange, config]
  );

  return (
    <>
      <FormControl>
        <FormLabel m={0}>{t('controlLayers.filter.alpha_to_outline.line_width')}</FormLabel>
        <CompositeSlider
          value={config.line_width}
          onChange={handleLineWidthChange}
          defaultValue={DEFAULTS.line_width}
          min={1}
          max={256}
        />
        <CompositeNumberInput
          value={config.line_width}
          onChange={handleLineWidthChange}
          defaultValue={DEFAULTS.line_width}
          min={1}
          max={256}
        />
      </FormControl>
    </>
  );
};

FilterAlphaToOutline.displayName = 'FilterAlphaToOutline';
