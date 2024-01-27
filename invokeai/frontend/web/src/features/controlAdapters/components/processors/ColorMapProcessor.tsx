import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useProcessorNodeChanged } from 'features/controlAdapters/components/hooks/useProcessorNodeChanged';
import { CONTROLNET_PROCESSORS } from 'features/controlAdapters/store/constants';
import type { RequiredColorMapImageProcessorInvocation } from 'features/controlAdapters/store/types';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import ProcessorWrapper from './common/ProcessorWrapper';

const DEFAULTS = CONTROLNET_PROCESSORS.color_map_image_processor.default as RequiredColorMapImageProcessorInvocation;

type ColorMapProcessorProps = {
  controlNetId: string;
  processorNode: RequiredColorMapImageProcessorInvocation;
  isEnabled: boolean;
};

const ColorMapProcessor = (props: ColorMapProcessorProps) => {
  const { controlNetId, processorNode, isEnabled } = props;
  const { color_map_tile_size } = processorNode;
  const processorChanged = useProcessorNodeChanged();
  const { t } = useTranslation();

  const handleColorMapTileSizeChanged = useCallback(
    (v: number) => {
      processorChanged(controlNetId, { color_map_tile_size: v });
    },
    [controlNetId, processorChanged]
  );

  return (
    <ProcessorWrapper>
      <FormControl isDisabled={!isEnabled}>
        <FormLabel>{t('controlnet.colorMapTileSize')}</FormLabel>
        <CompositeSlider
          value={color_map_tile_size}
          defaultValue={DEFAULTS.color_map_tile_size}
          onChange={handleColorMapTileSizeChanged}
          min={1}
          max={256}
          step={1}
          marks
        />
        <CompositeNumberInput
          value={color_map_tile_size}
          defaultValue={DEFAULTS.color_map_tile_size}
          onChange={handleColorMapTileSizeChanged}
          min={1}
          max={4096}
          step={1}
        />
      </FormControl>
    </ProcessorWrapper>
  );
};

export default memo(ColorMapProcessor);
