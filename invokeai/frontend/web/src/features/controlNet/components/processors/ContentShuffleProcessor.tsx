import IAISlider from 'common/components/IAISlider';
import { CONTROLNET_PROCESSORS } from 'features/controlNet/store/constants';
import { RequiredContentShuffleImageProcessorInvocation } from 'features/controlNet/store/types';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useProcessorNodeChanged } from '../hooks/useProcessorNodeChanged';
import ProcessorWrapper from './common/ProcessorWrapper';

const DEFAULTS = CONTROLNET_PROCESSORS.content_shuffle_image_processor
  .default as RequiredContentShuffleImageProcessorInvocation;

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

  const handleDetectResolutionChanged = useCallback(
    (v: number) => {
      processorChanged(controlNetId, { detect_resolution: v });
    },
    [controlNetId, processorChanged]
  );

  const handleDetectResolutionReset = useCallback(() => {
    processorChanged(controlNetId, {
      detect_resolution: DEFAULTS.detect_resolution,
    });
  }, [controlNetId, processorChanged]);

  const handleImageResolutionChanged = useCallback(
    (v: number) => {
      processorChanged(controlNetId, { image_resolution: v });
    },
    [controlNetId, processorChanged]
  );

  const handleImageResolutionReset = useCallback(() => {
    processorChanged(controlNetId, {
      image_resolution: DEFAULTS.image_resolution,
    });
  }, [controlNetId, processorChanged]);

  const handleWChanged = useCallback(
    (v: number) => {
      processorChanged(controlNetId, { w: v });
    },
    [controlNetId, processorChanged]
  );

  const handleWReset = useCallback(() => {
    processorChanged(controlNetId, {
      w: DEFAULTS.w,
    });
  }, [controlNetId, processorChanged]);

  const handleHChanged = useCallback(
    (v: number) => {
      processorChanged(controlNetId, { h: v });
    },
    [controlNetId, processorChanged]
  );

  const handleHReset = useCallback(() => {
    processorChanged(controlNetId, {
      h: DEFAULTS.h,
    });
  }, [controlNetId, processorChanged]);

  const handleFChanged = useCallback(
    (v: number) => {
      processorChanged(controlNetId, { f: v });
    },
    [controlNetId, processorChanged]
  );

  const handleFReset = useCallback(() => {
    processorChanged(controlNetId, {
      f: DEFAULTS.f,
    });
  }, [controlNetId, processorChanged]);

  return (
    <ProcessorWrapper>
      <IAISlider
        label={t('controlnet.detectResolution')}
        value={detect_resolution}
        onChange={handleDetectResolutionChanged}
        handleReset={handleDetectResolutionReset}
        withReset
        min={0}
        max={4096}
        withInput
        withSliderMarks
        isDisabled={!isEnabled}
      />
      <IAISlider
        label={t('controlnet.imageResolution')}
        value={image_resolution}
        onChange={handleImageResolutionChanged}
        handleReset={handleImageResolutionReset}
        withReset
        min={0}
        max={4096}
        withInput
        withSliderMarks
        isDisabled={!isEnabled}
      />
      <IAISlider
        label={t('controlnet.w')}
        value={w}
        onChange={handleWChanged}
        handleReset={handleWReset}
        withReset
        min={0}
        max={4096}
        withInput
        withSliderMarks
        isDisabled={!isEnabled}
      />
      <IAISlider
        label={t('controlnet.h')}
        value={h}
        onChange={handleHChanged}
        handleReset={handleHReset}
        withReset
        min={0}
        max={4096}
        withInput
        withSliderMarks
        isDisabled={!isEnabled}
      />
      <IAISlider
        label={t('controlnet.f')}
        value={f}
        onChange={handleFChanged}
        handleReset={handleFReset}
        withReset
        min={0}
        max={4096}
        withInput
        withSliderMarks
        isDisabled={!isEnabled}
      />
    </ProcessorWrapper>
  );
};

export default memo(ContentShuffleProcessor);
