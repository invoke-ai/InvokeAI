import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useProcessorNodeChanged } from 'features/controlAdapters/components/hooks/useProcessorNodeChanged';
import { useGetDefaultForControlnetProcessor } from 'features/controlAdapters/hooks/useGetDefaultForControlnetProcessor';
import type { RequiredColorMapImageProcessorInvocation } from 'features/controlAdapters/store/types';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import ProcessorWrapper from './common/ProcessorWrapper';

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
  const defaults = useGetDefaultForControlnetProcessor(
    'color_map_image_processor'
  ) as RequiredColorMapImageProcessorInvocation;

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
          defaultValue={defaults.color_map_tile_size}
          onChange={handleColorMapTileSizeChanged}
          min={1}
          max={256}
          step={1}
          marks
        />
        <CompositeNumberInput
          value={color_map_tile_size}
          defaultValue={defaults.color_map_tile_size}
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
