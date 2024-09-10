import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import type { ColorMapProcessorConfig } from 'features/controlLayers/store/types';
import { IMAGE_FILTERS } from 'features/controlLayers/store/types';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import type { FilterComponentProps } from './types';

type Props = FilterComponentProps<ColorMapProcessorConfig>;
const DEFAULTS = IMAGE_FILTERS['color_map_image_processor'].buildDefaults();

export const FilterColorMap = memo(({ onChange, config }: Props) => {
  const { t } = useTranslation();
  const handleColorMapTileSizeChanged = useCallback(
    (v: number) => {
      onChange({ ...config, color_map_tile_size: v });
    },
    [config, onChange]
  );

  return (
    <>
      <FormControl>
        <FormLabel m={0}>{t('controlnet.colorMapTileSize')}</FormLabel>
        <CompositeSlider
          value={config.color_map_tile_size}
          defaultValue={DEFAULTS.color_map_tile_size}
          onChange={handleColorMapTileSizeChanged}
          min={1}
          max={256}
          step={1}
          marks
        />
        <CompositeNumberInput
          value={config.color_map_tile_size}
          defaultValue={DEFAULTS.color_map_tile_size}
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
