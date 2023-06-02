import { Flex } from '@chakra-ui/react';
import IAISlider from 'common/components/IAISlider';
import { memo, useCallback } from 'react';
import { useProcessorNodeChanged } from '../hooks/useProcessorNodeChanged';
import { RequiredCannyImageProcessorInvocation } from 'features/controlNet/store/types';
import { CONTROLNET_PROCESSORS } from 'features/controlNet/store/constants';

const DEFAULTS = CONTROLNET_PROCESSORS.canny_image_processor.default;

type CannyProcessorProps = {
  controlNetId: string;
  processorNode: RequiredCannyImageProcessorInvocation;
};

const CannyProcessor = (props: CannyProcessorProps) => {
  const { controlNetId, processorNode } = props;
  const { low_threshold, high_threshold } = processorNode;
  const processorChanged = useProcessorNodeChanged();

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
    <Flex sx={{ flexDirection: 'column', gap: 2 }}>
      <IAISlider
        label="Low Threshold"
        value={low_threshold}
        onChange={handleLowThresholdChanged}
        handleReset={handleLowThresholdReset}
        withReset
        min={0}
        max={255}
        withInput
      />
      <IAISlider
        label="High Threshold"
        value={high_threshold}
        onChange={handleHighThresholdChanged}
        handleReset={handleHighThresholdReset}
        withReset
        min={0}
        max={255}
        withInput
      />
    </Flex>
  );
};

export default memo(CannyProcessor);
