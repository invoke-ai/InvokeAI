import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { useProcessorNodeChanged } from 'features/controlAdapters/components/hooks/useProcessorNodeChanged';
import { CONTROLNET_PROCESSORS } from 'features/controlAdapters/store/constants';
import type { RequiredMediapipeFaceProcessorInvocation } from 'features/controlAdapters/store/types';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

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
      <InvControl label={t('controlnet.maxFaces')} isDisabled={!isEnabled}>
        <InvSlider
          value={max_faces}
          onChange={handleMaxFacesChanged}
          onReset={handleMaxFacesReset}
          min={1}
          max={20}
          marks
          withNumberInput
        />
      </InvControl>
      <InvControl label={t('controlnet.minConfidence')} isDisabled={!isEnabled}>
        <InvSlider
          value={min_confidence}
          onChange={handleMinConfidenceChanged}
          onReset={handleMinConfidenceReset}
          min={0}
          max={1}
          step={0.01}
          marks
          withNumberInput
        />
      </InvControl>
    </ProcessorWrapper>
  );
};

export default memo(MediapipeFaceProcessor);
