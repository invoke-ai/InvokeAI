import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useProcessorNodeChanged } from 'features/controlAdapters/components/hooks/useProcessorNodeChanged';
import { useGetDefaultForControlnetProcessor } from 'features/controlAdapters/hooks/useGetDefaultForControlnetProcessor';
import type { RequiredMediapipeFaceProcessorInvocation } from 'features/controlAdapters/store/types';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import ProcessorWrapper from './common/ProcessorWrapper';

type Props = {
  controlNetId: string;
  processorNode: RequiredMediapipeFaceProcessorInvocation;
  isEnabled: boolean;
};

const MediapipeFaceProcessor = (props: Props) => {
  const { controlNetId, processorNode, isEnabled } = props;
  const { max_faces, min_confidence, image_resolution, detect_resolution } = processorNode;
  const processorChanged = useProcessorNodeChanged();
  const { t } = useTranslation();

  const defaults = useGetDefaultForControlnetProcessor(
    'mediapipe_face_processor'
  ) as RequiredMediapipeFaceProcessorInvocation;

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

  const handleImageResolutionChanged = useCallback(
    (v: number) => {
      processorChanged(controlNetId, { image_resolution: v });
    },
    [controlNetId, processorChanged]
  );

  const handleDetectResolutionChanged = useCallback(
    (v: number) => {
      processorChanged(controlNetId, { detect_resolution: v });
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
          defaultValue={defaults.max_faces}
          min={1}
          max={20}
          marks
        />
        <CompositeNumberInput
          value={max_faces}
          onChange={handleMaxFacesChanged}
          defaultValue={defaults.max_faces}
          min={1}
          max={20}
        />
      </FormControl>
      <FormControl isDisabled={!isEnabled}>
        <FormLabel>{t('controlnet.minConfidence')}</FormLabel>
        <CompositeSlider
          value={min_confidence}
          onChange={handleMinConfidenceChanged}
          defaultValue={defaults.min_confidence}
          min={0}
          max={1}
          step={0.01}
          marks
        />
        <CompositeNumberInput
          value={min_confidence}
          onChange={handleMinConfidenceChanged}
          defaultValue={defaults.min_confidence}
          min={0}
          max={1}
          step={0.01}
        />
      </FormControl>
      <FormControl isDisabled={!isEnabled}>
        <FormLabel>{t('controlnet.imageResolution')}</FormLabel>
        <CompositeSlider
          value={image_resolution}
          onChange={handleImageResolutionChanged}
          defaultValue={defaults.image_resolution}
          min={0}
          max={4096}
          marks
        />
        <CompositeNumberInput
          value={image_resolution}
          onChange={handleImageResolutionChanged}
          defaultValue={defaults.image_resolution}
          min={0}
          max={4096}
        />
      </FormControl>
      <FormControl isDisabled={!isEnabled}>
        <FormLabel>{t('controlnet.detectResolution')}</FormLabel>
        <CompositeSlider
          value={detect_resolution}
          onChange={handleDetectResolutionChanged}
          defaultValue={defaults.detect_resolution}
          min={0}
          max={4096}
          marks
        />
        <CompositeNumberInput
          value={detect_resolution}
          onChange={handleDetectResolutionChanged}
          defaultValue={defaults.detect_resolution}
          min={0}
          max={4096}
        />
      </FormControl>
    </ProcessorWrapper>
  );
};

export default memo(MediapipeFaceProcessor);
