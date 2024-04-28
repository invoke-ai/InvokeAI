import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useProcessorNodeChanged } from 'features/controlAdapters/components/hooks/useProcessorNodeChanged';
import { useGetDefaultForControlnetProcessor } from 'features/controlAdapters/hooks/useGetDefaultForControlnetProcessor';
import type { RequiredCannyImageProcessorInvocation } from 'features/controlAdapters/store/types';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import ProcessorWrapper from './common/ProcessorWrapper';

type CannyProcessorProps = {
  controlNetId: string;
  processorNode: RequiredCannyImageProcessorInvocation;
  isEnabled: boolean;
};

const CannyProcessor = (props: CannyProcessorProps) => {
  const { controlNetId, processorNode, isEnabled } = props;
  const { low_threshold, high_threshold, image_resolution, detect_resolution } = processorNode;
  const processorChanged = useProcessorNodeChanged();
  const { t } = useTranslation();
  const defaults = useGetDefaultForControlnetProcessor(
    'canny_image_processor'
  ) as RequiredCannyImageProcessorInvocation;

  const handleLowThresholdChanged = useCallback(
    (v: number) => {
      processorChanged(controlNetId, { low_threshold: v });
    },
    [controlNetId, processorChanged]
  );

  const handleHighThresholdChanged = useCallback(
    (v: number) => {
      processorChanged(controlNetId, { high_threshold: v });
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
        <FormLabel>{t('controlnet.lowThreshold')}</FormLabel>
        <CompositeSlider
          value={low_threshold}
          onChange={handleLowThresholdChanged}
          defaultValue={defaults.low_threshold}
          min={0}
          max={255}
        />
        <CompositeNumberInput
          value={low_threshold}
          onChange={handleLowThresholdChanged}
          defaultValue={defaults.low_threshold}
          min={0}
          max={255}
        />
      </FormControl>
      <FormControl isDisabled={!isEnabled}>
        <FormLabel>{t('controlnet.highThreshold')}</FormLabel>
        <CompositeSlider
          value={high_threshold}
          onChange={handleHighThresholdChanged}
          defaultValue={defaults.high_threshold}
          min={0}
          max={255}
        />
        <CompositeNumberInput
          value={high_threshold}
          onChange={handleHighThresholdChanged}
          defaultValue={defaults.high_threshold}
          min={0}
          max={255}
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

export default memo(CannyProcessor);
