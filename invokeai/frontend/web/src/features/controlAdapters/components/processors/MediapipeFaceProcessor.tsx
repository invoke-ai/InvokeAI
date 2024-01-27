import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useProcessorNodeChanged } from 'features/controlAdapters/components/hooks/useProcessorNodeChanged';
import { CONTROLNET_PROCESSORS } from 'features/controlAdapters/store/constants';
import type { RequiredMediapipeFaceProcessorInvocation } from 'features/controlAdapters/store/types';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import ProcessorWrapper from './common/ProcessorWrapper';

const DEFAULTS = CONTROLNET_PROCESSORS.mediapipe_face_processor.default as RequiredMediapipeFaceProcessorInvocation;

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

  return (
    <ProcessorWrapper>
      <FormControl isDisabled={!isEnabled}>
        <FormLabel>{t('controlnet.maxFaces')}</FormLabel>
        <CompositeSlider
          value={max_faces}
          onChange={handleMaxFacesChanged}
          defaultValue={DEFAULTS.max_faces}
          min={1}
          max={20}
          marks
        />
        <CompositeNumberInput
          value={max_faces}
          onChange={handleMaxFacesChanged}
          defaultValue={DEFAULTS.max_faces}
          min={1}
          max={20}
        />
      </FormControl>
      <FormControl isDisabled={!isEnabled}>
        <FormLabel>{t('controlnet.minConfidence')}</FormLabel>
        <CompositeSlider
          value={min_confidence}
          onChange={handleMinConfidenceChanged}
          defaultValue={DEFAULTS.min_confidence}
          min={0}
          max={1}
          step={0.01}
          marks
        />
        <CompositeNumberInput
          value={min_confidence}
          onChange={handleMinConfidenceChanged}
          defaultValue={DEFAULTS.min_confidence}
          min={0}
          max={1}
          step={0.01}
        />
      </FormControl>
    </ProcessorWrapper>
  );
};

export default memo(MediapipeFaceProcessor);
