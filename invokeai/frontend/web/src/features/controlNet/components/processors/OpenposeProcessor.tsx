import { Flex } from '@chakra-ui/react';
import IAISlider from 'common/components/IAISlider';
import { ChangeEvent, memo, useCallback } from 'react';
import { useProcessorNodeChanged } from '../hooks/useProcessorNodeChanged';
import { RequiredOpenposeImageProcessorInvocation } from 'features/controlNet/store/types';
import IAISwitch from 'common/components/IAISwitch';

type Props = {
  controlNetId: string;
  processorNode: RequiredOpenposeImageProcessorInvocation;
};

const OpenposeProcessor = (props: Props) => {
  const { controlNetId, processorNode } = props;
  const { image_resolution, detect_resolution, hand_and_face } = processorNode;
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

  const handleHandAndFaceChanged = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      processorChanged(controlNetId, { hand_and_face: e.target.checked });
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
      <IAISwitch
        label="Hand and Face"
        isChecked={hand_and_face}
        onChange={handleHandAndFaceChanged}
      />
    </Flex>
  );
};

export default memo(OpenposeProcessor);
