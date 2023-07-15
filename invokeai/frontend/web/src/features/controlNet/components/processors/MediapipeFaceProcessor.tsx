import { useAppSelector } from 'app/store/storeHooks';
import IAISlider from 'common/components/IAISlider';
import { CONTROLNET_PROCESSORS } from 'features/controlNet/store/constants';
import { RequiredMediapipeFaceProcessorInvocation } from 'features/controlNet/store/types';
import { selectIsBusy } from 'features/system/store/systemSelectors';
import { memo, useCallback } from 'react';
import { useProcessorNodeChanged } from '../hooks/useProcessorNodeChanged';
import ProcessorWrapper from './common/ProcessorWrapper';

const DEFAULTS = CONTROLNET_PROCESSORS.mediapipe_face_processor
  .default as RequiredMediapipeFaceProcessorInvocation;

type Props = {
  controlNetId: string;
  processorNode: RequiredMediapipeFaceProcessorInvocation;
  isEnabled: boolean;
};

const MediapipeFaceProcessor = (props: Props) => {
  const { controlNetId, processorNode, isEnabled } = props;
  const { max_faces, min_confidence } = processorNode;
  const processorChanged = useProcessorNodeChanged();
  const isBusy = useAppSelector(selectIsBusy);

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

  const handleMaxFacesReset = useCallback(() => {
    processorChanged(controlNetId, { max_faces: DEFAULTS.max_faces });
  }, [controlNetId, processorChanged]);

  const handleMinConfidenceReset = useCallback(() => {
    processorChanged(controlNetId, { min_confidence: DEFAULTS.min_confidence });
  }, [controlNetId, processorChanged]);

  return (
    <ProcessorWrapper>
      <IAISlider
        label="Max Faces"
        value={max_faces}
        onChange={handleMaxFacesChanged}
        handleReset={handleMaxFacesReset}
        withReset
        min={1}
        max={20}
        withInput
        withSliderMarks
        isDisabled={isBusy || !isEnabled}
      />
      <IAISlider
        label="Min Confidence"
        value={min_confidence}
        onChange={handleMinConfidenceChanged}
        handleReset={handleMinConfidenceReset}
        withReset
        min={0}
        max={1}
        step={0.01}
        withInput
        withSliderMarks
        isDisabled={isBusy || !isEnabled}
      />
    </ProcessorWrapper>
  );
};

export default memo(MediapipeFaceProcessor);
