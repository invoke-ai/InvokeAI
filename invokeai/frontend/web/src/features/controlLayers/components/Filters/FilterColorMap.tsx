import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { type ColorMapFilterConfig, IMAGE_FILTERS } from 'features/controlLayers/store/filters';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import type { FilterComponentProps } from './types';

type Props = FilterComponentProps<ColorMapFilterConfig>;
const DEFAULTS = IMAGE_FILTERS.color_map.buildDefaults();

export const FilterColorMap = memo(({ onChange, config }: Props) => {
  const { t } = useTranslation();
  const handleColorMapTileSizeChanged = useCallback(
    (v: number) => {
      onChange({ ...config, tile_size: v });
    },
    [config, onChange]
  );

  return (
    <>
      <FormControl>
        <FormLabel m={0}>{t('controlLayers.filter.color_map.tile_size')}</FormLabel>
        <CompositeSlider
          value={config.tile_size}
          defaultValue={DEFAULTS.tile_size}
          onChange={handleColorMapTileSizeChanged}
          min={1}
          max={256}
          step={1}
          marks
        />
        <CompositeNumberInput
          value={config.tile_size}
          defaultValue={DEFAULTS.tile_size}
          onChange={handleColorMapTileSizeChanged}
          min={1}
          max={4096}
          step={1}
        />
      </FormControl>
    </>
  );
});

FilterColorMap.displayName = 'FilterColorMap';
