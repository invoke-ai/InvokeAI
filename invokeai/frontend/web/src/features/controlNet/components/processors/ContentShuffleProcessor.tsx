import { Flex } from '@chakra-ui/react';
import IAISlider from 'common/components/IAISlider';
import { memo, useCallback } from 'react';
import { useProcessorNodeChanged } from '../hooks/useProcessorNodeChanged';
import { RequiredContentShuffleImageProcessorInvocation } from 'features/controlNet/store/types';

type Props = {
  controlNetId: string;
  processorNode: RequiredContentShuffleImageProcessorInvocation;
};

const ContentShuffleProcessor = (props: Props) => {
  const { controlNetId, processorNode } = props;
  const { image_resolution, detect_resolution, w, h, f } = processorNode;
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

  const handleWChanged = useCallback(
    (v: number) => {
      processorChanged(controlNetId, { w: v });
    },
    [controlNetId, processorChanged]
  );

  const handleHChanged = useCallback(
    (v: number) => {
      processorChanged(controlNetId, { h: v });
    },
    [controlNetId, processorChanged]
  );

  const handleFChanged = useCallback(
    (v: number) => {
      processorChanged(controlNetId, { f: v });
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
        value={w}
        onChange={handleWChanged}
        min={0}
        max={4096}
        withInput
      />
      <IAISlider
        label="H"
        value={h}
        onChange={handleHChanged}
        min={0}
        max={4096}
        withInput
      />
      <IAISlider
        label="F"
        value={f}
        onChange={handleFChanged}
        min={0}
        max={4096}
        withInput
      />
    </Flex>
  );
};

export default memo(ContentShuffleProcessor);
