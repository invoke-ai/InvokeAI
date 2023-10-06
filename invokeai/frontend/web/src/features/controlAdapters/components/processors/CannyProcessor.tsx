import IAISlider from 'common/components/IAISlider';
import { CONTROLNET_PROCESSORS } from 'features/controlAdapters/store/constants';
import { RequiredCannyImageProcessorInvocation } from 'features/controlAdapters/store/types';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useProcessorNodeChanged } from '../hooks/useProcessorNodeChanged';
import ProcessorWrapper from './common/ProcessorWrapper';

const DEFAULTS = CONTROLNET_PROCESSORS.canny_image_processor
  .default as RequiredCannyImageProcessorInvocation;

type CannyProcessorProps = {
  controlNetId: string;
  processorNode: RequiredCannyImageProcessorInvocation;
  isEnabled: boolean;
};

const CannyProcessor = (props: CannyProcessorProps) => {
  const { controlNetId, processorNode, isEnabled } = props;
  const { low_threshold, high_threshold } = processorNode;
  const processorChanged = useProcessorNodeChanged();
  const { t } = useTranslation();

  const handleLowThresholdChanged = useCallback(
    (v: number) => {
      processorChanged(controlNetId, { low_threshold: v });
    },
    [controlNetId, processorChanged]
  );

  const handleLowThresholdReset = useCallback(() => {
    processorChanged(controlNetId, {
      low_threshold: DEFAULTS.low_threshold,
    });
  }, [controlNetId, processorChanged]);

  const handleHighThresholdChanged = useCallback(
    (v: number) => {
      processorChanged(controlNetId, { high_threshold: v });
    },
    [controlNetId, processorChanged]
  );

  const handleHighThresholdReset = useCallback(() => {
    processorChanged(controlNetId, {
      high_threshold: DEFAULTS.high_threshold,
    });
  }, [controlNetId, processorChanged]);

  return (
    <ProcessorWrapper>
      <IAISlider
        isDisabled={!isEnabled}
        label={t('controlnet.lowThreshold')}
        value={low_threshold}
        onChange={handleLowThresholdChanged}
        handleReset={handleLowThresholdReset}
        withReset
        min={0}
        max={255}
        withInput
        withSliderMarks
      />
      <IAISlider
        isDisabled={!isEnabled}
        label={t('controlnet.highThreshold')}
        value={high_threshold}
        onChange={handleHighThresholdChanged}
        handleReset={handleHighThresholdReset}
        withReset
        min={0}
        max={255}
        withInput
        withSliderMarks
      />
    </ProcessorWrapper>
  );
};

export default memo(CannyProcessor);
