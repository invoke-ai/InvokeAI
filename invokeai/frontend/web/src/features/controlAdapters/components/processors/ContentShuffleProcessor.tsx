import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useProcessorNodeChanged } from 'features/controlAdapters/components/hooks/useProcessorNodeChanged';
import { useGetDefaultForControlnetProcessor } from 'features/controlAdapters/hooks/useGetDefaultForControlnetProcessor';
import type { RequiredContentShuffleImageProcessorInvocation } from 'features/controlAdapters/store/types';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import ProcessorWrapper from './common/ProcessorWrapper';

type Props = {
  controlNetId: string;
  processorNode: RequiredContentShuffleImageProcessorInvocation;
  isEnabled: boolean;
};

const ContentShuffleProcessor = (props: Props) => {
  const { controlNetId, processorNode, isEnabled } = props;
  const { image_resolution, detect_resolution, w, h, f } = processorNode;
  const processorChanged = useProcessorNodeChanged();
  const { t } = useTranslation();

  const defaults = useGetDefaultForControlnetProcessor(
    'content_shuffle_image_processor'
  ) as RequiredContentShuffleImageProcessorInvocation;

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

  const handleWChanged = useCallback(
    (v: number) => {
      processorChanged(controlNetId, { w: v });
    },
    [controlNetId, processorChanged]
  );

  const handleHChanged = useCallback(
    (v: number) => {
      processorChanged(controlNetId, { h: v });
    },
    [controlNetId, processorChanged]
  );

  const handleFChanged = useCallback(
    (v: number) => {
      processorChanged(controlNetId, { f: v });
    },
    [controlNetId, processorChanged]
  );

  return (
    <ProcessorWrapper>
      <FormControl isDisabled={!isEnabled}>
        <FormLabel>{t('controlnet.detectResolution')}</FormLabel>
        <CompositeSlider
          value={detect_resolution}
          defaultValue={defaults.detect_resolution}
          onChange={handleDetectResolutionChanged}
          min={0}
          max={4096}
          marks
        />
        <CompositeNumberInput
          value={detect_resolution}
          defaultValue={defaults.detect_resolution}
          onChange={handleDetectResolutionChanged}
          min={0}
          max={4096}
        />
      </FormControl>
      <FormControl isDisabled={!isEnabled}>
        <FormLabel>{t('controlnet.imageResolution')}</FormLabel>
        <CompositeSlider
          value={image_resolution}
          defaultValue={defaults.image_resolution}
          onChange={handleImageResolutionChanged}
          min={0}
          max={4096}
          marks
        />
        <CompositeNumberInput
          value={image_resolution}
          defaultValue={defaults.image_resolution}
          onChange={handleImageResolutionChanged}
          min={0}
          max={4096}
        />
      </FormControl>
      <FormControl isDisabled={!isEnabled}>
        <FormLabel>{t('controlnet.w')}</FormLabel>
        <CompositeSlider value={w} defaultValue={defaults.w} onChange={handleWChanged} min={0} max={4096} marks />
        <CompositeNumberInput value={w} defaultValue={defaults.w} onChange={handleWChanged} min={0} max={4096} />
      </FormControl>
      <FormControl isDisabled={!isEnabled}>
        <FormLabel>{t('controlnet.h')}</FormLabel>
        <CompositeSlider value={h} defaultValue={defaults.h} onChange={handleHChanged} min={0} max={4096} marks />
        <CompositeNumberInput value={h} defaultValue={defaults.h} onChange={handleHChanged} min={0} max={4096} />
      </FormControl>
      <FormControl isDisabled={!isEnabled}>
        <FormLabel>{t('controlnet.f')}</FormLabel>
        <CompositeSlider value={f} defaultValue={defaults.f} onChange={handleFChanged} min={0} max={4096} marks />
        <CompositeNumberInput value={f} defaultValue={defaults.f} onChange={handleFChanged} min={0} max={4096} />
      </FormControl>
    </ProcessorWrapper>
  );
};

export default memo(ContentShuffleProcessor);
