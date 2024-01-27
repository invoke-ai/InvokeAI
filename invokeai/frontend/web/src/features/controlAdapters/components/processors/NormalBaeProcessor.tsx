import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useProcessorNodeChanged } from 'features/controlAdapters/components/hooks/useProcessorNodeChanged';
import { CONTROLNET_PROCESSORS } from 'features/controlAdapters/store/constants';
import type { RequiredNormalbaeImageProcessorInvocation } from 'features/controlAdapters/store/types';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import ProcessorWrapper from './common/ProcessorWrapper';

const DEFAULTS = CONTROLNET_PROCESSORS.normalbae_image_processor.default as RequiredNormalbaeImageProcessorInvocation;

type Props = {
  controlNetId: string;
  processorNode: RequiredNormalbaeImageProcessorInvocation;
  isEnabled: boolean;
};

const NormalBaeProcessor = (props: Props) => {
  const { controlNetId, processorNode, isEnabled } = props;
  const { image_resolution, detect_resolution } = processorNode;
  const processorChanged = useProcessorNodeChanged();
  const { t } = useTranslation();

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

  return (
    <ProcessorWrapper>
      <FormControl isDisabled={!isEnabled}>
        <FormLabel>{t('controlnet.detectResolution')}</FormLabel>
        <CompositeSlider
          value={detect_resolution}
          onChange={handleDetectResolutionChanged}
          defaultValue={DEFAULTS.detect_resolution}
          min={0}
          max={4096}
          marks
        />
        <CompositeNumberInput
          value={detect_resolution}
          onChange={handleDetectResolutionChanged}
          defaultValue={DEFAULTS.detect_resolution}
          min={0}
          max={4096}
        />
      </FormControl>
      <FormControl isDisabled={!isEnabled}>
        <FormLabel>{t('controlnet.imageResolution')}</FormLabel>
        <CompositeSlider
          value={image_resolution}
          onChange={handleImageResolutionChanged}
          defaultValue={DEFAULTS.image_resolution}
          min={0}
          max={4096}
          marks
        />
        <CompositeNumberInput
          value={image_resolution}
          onChange={handleImageResolutionChanged}
          defaultValue={DEFAULTS.image_resolution}
          min={0}
          max={4096}
        />
      </FormControl>
    </ProcessorWrapper>
  );
};

export default memo(NormalBaeProcessor);
