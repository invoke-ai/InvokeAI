import { Flex } from '@chakra-ui/react';
import IAISlider from 'common/components/IAISlider';
import { memo, useCallback } from 'react';
import { useProcessorNodeChanged } from '../hooks/useProcessorNodeChanged';
import { RequiredMlsdImageProcessorInvocation } from 'features/controlNet/store/types';

type Props = {
  controlNetId: string;
  processorNode: RequiredMlsdImageProcessorInvocation;
};

const MlsdImageProcessor = (props: Props) => {
  const { controlNetId, processorNode } = props;
  const { image_resolution, detect_resolution, thr_d, thr_v } = processorNode;
  const processorChanged = useProcessorNodeChanged();

  const handleDetectResolutionChanged = useCallback(
    (v: number) => {
      processorChanged(controlNetId, { detect_resolution: v });
    },
    [controlNetId, processorChanged]
  );

  const handleImageResolutionChanged = useCallback(
    (v: number) => {
      processorChanged(controlNetId, { image_resolution: v });
    },
    [controlNetId, processorChanged]
  );

  const handleThrDChanged = useCallback(
    (v: number) => {
      processorChanged(controlNetId, { thr_d: v });
    },
    [controlNetId, processorChanged]
  );

  const handleThrVChanged = useCallback(
    (v: number) => {
      processorChanged(controlNetId, { thr_v: v });
    },
    [controlNetId, processorChanged]
  );

  return (
    <Flex sx={{ flexDirection: 'column', gap: 2 }}>
      <IAISlider
        label="Detect Resolution"
        value={detect_resolution}
        onChange={handleDetectResolutionChanged}
        min={0}
        max={4096}
        withInput
      />
      <IAISlider
        label="Image Resolution"
        value={image_resolution}
        onChange={handleImageResolutionChanged}
        min={0}
        max={4096}
        withInput
      />
      <IAISlider
        label="W"
        value={thr_d}
        onChange={handleThrDChanged}
        min={0}
        max={1}
        step={0.01}
        withInput
      />
      <IAISlider
        label="H"
        value={thr_v}
        onChange={handleThrVChanged}
        min={0}
        max={1}
        step={0.01}
        withInput
      />
    </Flex>
  );
};

export default memo(MlsdImageProcessor);
