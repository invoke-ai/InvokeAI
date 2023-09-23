import IAISlider from 'common/components/IAISlider';
import { CONTROLNET_PROCESSORS } from 'features/controlNet/store/constants';
import { RequiredColorMapImageProcessorInvocation } from 'features/controlNet/store/types';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useProcessorNodeChanged } from '../hooks/useProcessorNodeChanged';
import ProcessorWrapper from './common/ProcessorWrapper';

const DEFAULTS = CONTROLNET_PROCESSORS.color_map_image_processor
  .default as RequiredColorMapImageProcessorInvocation;

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

  const handleColorMapTileSizeReset = useCallback(() => {
    processorChanged(controlNetId, {
      color_map_tile_size: DEFAULTS.color_map_tile_size,
    });
  }, [controlNetId, processorChanged]);

  return (
    <ProcessorWrapper>
      <IAISlider
        isDisabled={!isEnabled}
        label={t('controlnet.colorMapTileSize')}
        value={color_map_tile_size}
        onChange={handleColorMapTileSizeChanged}
        handleReset={handleColorMapTileSizeReset}
        withReset
        min={1}
        max={256}
        step={1}
        withInput
        withSliderMarks
        sliderNumberInputProps={{
          max: 4096,
        }}
      />
    </ProcessorWrapper>
  );
};

export default memo(ColorMapProcessor);
