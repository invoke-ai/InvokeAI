import IAISlider from 'common/components/IAISlider';
import { CONTROLNET_PROCESSORS } from 'features/controlAdapters/store/constants';
import { RequiredMediapipeFaceProcessorInvocation } from 'features/controlAdapters/store/types';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useProcessorNodeChanged } from 'features/controlAdapters/components/hooks/useProcessorNodeChanged';
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
  const { t } = useTranslation();

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
        label={t('controlnet.maxFaces')}
        value={max_faces}
        onChange={handleMaxFacesChanged}
        handleReset={handleMaxFacesReset}
        withReset
        min={1}
        max={20}
        withInput
        withSliderMarks
        isDisabled={!isEnabled}
      />
      <IAISlider
        label={t('controlnet.minConfidence')}
        value={min_confidence}
        onChange={handleMinConfidenceChanged}
        handleReset={handleMinConfidenceReset}
        withReset
        min={0}
        max={1}
        step={0.01}
        withInput
        withSliderMarks
        isDisabled={!isEnabled}
      />
    </ProcessorWrapper>
  );
};

export default memo(MediapipeFaceProcessor);
