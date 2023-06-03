import IAISlider from 'common/components/IAISlider';
import { CONTROLNET_PROCESSORS } from 'features/controlNet/store/constants';
import { RequiredMidasDepthImageProcessorInvocation } from 'features/controlNet/store/types';
import { memo, useCallback } from 'react';
import { useProcessorNodeChanged } from '../hooks/useProcessorNodeChanged';
import ProcessorOptionsContainer from './shared/ProcessorOptionsContainer';

const DEFAULTS = CONTROLNET_PROCESSORS.midas_depth_image_processor.default;

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

  const handleAMultReset = useCallback(() => {
    processorChanged(controlNetId, { a_mult: DEFAULTS.a_mult });
  }, [controlNetId, processorChanged]);

  const handleBgThReset = useCallback(() => {
    processorChanged(controlNetId, { bg_th: DEFAULTS.bg_th });
  }, [controlNetId, processorChanged]);

  return (
    <ProcessorOptionsContainer>
      <IAISlider
        label="a_mult"
        value={a_mult}
        onChange={handleAMultChanged}
        handleReset={handleAMultReset}
        withReset
        min={0}
        max={20}
        step={0.01}
        withInput
      />
      <IAISlider
        label="bg_th"
        value={bg_th}
        onChange={handleBgThChanged}
        handleReset={handleBgThReset}
        withReset
        min={0}
        max={20}
        step={0.01}
        withInput
      />
    </ProcessorOptionsContainer>
  );
};

export default memo(MidasDepthProcessor);
