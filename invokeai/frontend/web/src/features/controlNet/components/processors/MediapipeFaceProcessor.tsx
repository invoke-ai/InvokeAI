import { Flex } from '@chakra-ui/react';
import IAISlider from 'common/components/IAISlider';
import { memo, useCallback } from 'react';
import { useProcessorNodeChanged } from '../hooks/useProcessorNodeChanged';
import {
  RequiredContentShuffleImageProcessorInvocation,
  RequiredMediapipeFaceProcessorInvocation,
} from 'features/controlNet/store/types';

type Props = {
  controlNetId: string;
  processorNode: RequiredMediapipeFaceProcessorInvocation;
};

const MediapipeFaceProcessor = (props: Props) => {
  const { controlNetId, processorNode } = props;
  const { max_faces, min_confidence } = processorNode;
  const processorChanged = useProcessorNodeChanged();

  const handleMaxFacesChanged = useCallback(
    (v: number) => {
      processorChanged(controlNetId, { max_faces: v });
    },
    [controlNetId, processorChanged]
  );

  const handleMinConfidenceChanged = useCallback(
    (v: number) => {
      processorChanged(controlNetId, { min_confidence: v });
    },
    [controlNetId, processorChanged]
  );

  return (
    <Flex sx={{ flexDirection: 'column', gap: 2 }}>
      <IAISlider
        label="Max Faces"
        value={max_faces}
        onChange={handleMaxFacesChanged}
        min={1}
        max={20}
        withInput
      />
      <IAISlider
        label="Min Confidence"
        value={min_confidence}
        onChange={handleMinConfidenceChanged}
        min={0}
        max={1}
        step={0.01}
        withInput
      />
    </Flex>
  );
};

export default memo(MediapipeFaceProcessor);
