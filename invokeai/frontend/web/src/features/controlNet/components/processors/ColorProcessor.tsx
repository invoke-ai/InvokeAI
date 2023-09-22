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

const ColorProcessor = (props: ColorMapProcessorProps) => {
  const { controlNetId, processorNode, isEnabled } = props;
  const { map_resolution } = processorNode;
  const processorChanged = useProcessorNodeChanged();
  const { t } = useTranslation();

  const handleMapResolutionChanged = useCallback(
    (v: number) => {
      processorChanged(controlNetId, { map_resolution: v });
    },
    [controlNetId, processorChanged]
  );

  const handleMapResolutionReset = useCallback(() => {
    processorChanged(controlNetId, {
      map_resolution: DEFAULTS.map_resolution,
    });
  }, [controlNetId, processorChanged]);

  return (
    <ProcessorWrapper>
      <IAISlider
        isDisabled={!isEnabled}
        label={t('controlnet.mapResolution')}
        value={map_resolution}
        onChange={handleMapResolutionChanged}
        handleReset={handleMapResolutionReset}
        withReset
        min={64}
        max={2048}
        step={64}
        withInput
        withSliderMarks
      />
    </ProcessorWrapper>
  );
};

export default memo(ColorProcessor);
