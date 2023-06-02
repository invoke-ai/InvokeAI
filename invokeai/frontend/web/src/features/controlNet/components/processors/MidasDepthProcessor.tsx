import { Flex } from '@chakra-ui/react';
import IAISlider from 'common/components/IAISlider';
import { memo, useCallback } from 'react';
import { useProcessorNodeChanged } from '../hooks/useProcessorNodeChanged';
import { RequiredMidasDepthImageProcessorInvocation } from 'features/controlNet/store/types';

type Props = {
  controlNetId: string;
  processorNode: RequiredMidasDepthImageProcessorInvocation;
};

const MidasDepthProcessor = (props: Props) => {
  const { controlNetId, processorNode } = props;
  const { a_mult, bg_th } = processorNode;
  const processorChanged = useProcessorNodeChanged();

  const handleAMultChanged = useCallback(
    (v: number) => {
      processorChanged(controlNetId, { a_mult: v });
    },
    [controlNetId, processorChanged]
  );

  const handleBgThChanged = useCallback(
    (v: number) => {
      processorChanged(controlNetId, { bg_th: v });
    },
    [controlNetId, processorChanged]
  );

  return (
    <Flex sx={{ flexDirection: 'column', gap: 2 }}>
      <IAISlider
        label="a_mult"
        value={a_mult}
        onChange={handleAMultChanged}
        min={0}
        max={20}
        step={0.01}
        withInput
      />
      <IAISlider
        label="bg_th"
        value={bg_th}
        onChange={handleBgThChanged}
        min={0}
        max={20}
        step={0.01}
        withInput
      />
    </Flex>
  );
};

export default memo(MidasDepthProcessor);
