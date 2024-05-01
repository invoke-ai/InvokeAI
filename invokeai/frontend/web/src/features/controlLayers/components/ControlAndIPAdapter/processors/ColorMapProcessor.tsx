import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import type { ProcessorComponentProps } from 'features/controlLayers/components/ControlAndIPAdapter/processors/types';
import { type ColorMapProcessorConfig, CONTROLNET_PROCESSORS } from 'features/controlLayers/util/controlAdapters';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import ProcessorWrapper from './ProcessorWrapper';

type Props = ProcessorComponentProps<ColorMapProcessorConfig>;
const DEFAULTS = CONTROLNET_PROCESSORS['color_map_image_processor'].buildDefaults();

export const ColorMapProcessor = memo(({ onChange, config }: Props) => {
  const { t } = useTranslation();
  const handleColorMapTileSizeChanged = useCallback(
    (v: number) => {
      onChange({ ...config, color_map_tile_size: v });
    },
    [config, onChange]
  );

  return (
    <ProcessorWrapper>
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
    </ProcessorWrapper>
  );
});

ColorMapProcessor.displayName = 'ColorMapProcessor';
