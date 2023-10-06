import IAISlider from 'common/components/IAISlider';
import { CONTROLNET_PROCESSORS } from 'features/controlAdapters/store/constants';
import { RequiredMlsdImageProcessorInvocation } from 'features/controlAdapters/store/types';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useProcessorNodeChanged } from '../hooks/useProcessorNodeChanged';
import ProcessorWrapper from './common/ProcessorWrapper';

const DEFAULTS = CONTROLNET_PROCESSORS.mlsd_image_processor
  .default as RequiredMlsdImageProcessorInvocation;

type Props = {
  controlNetId: string;
  processorNode: RequiredMlsdImageProcessorInvocation;
  isEnabled: boolean;
};

const MlsdImageProcessor = (props: Props) => {
  const { controlNetId, processorNode, isEnabled } = props;
  const { image_resolution, detect_resolution, thr_d, thr_v } = processorNode;
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

  const handleThrDChanged = useCallback(
    (v: number) => {
      processorChanged(controlNetId, { thr_d: v });
    },
    [controlNetId, processorChanged]
  );

  const handleThrVChanged = useCallback(
    (v: number) => {
      processorChanged(controlNetId, { thr_v: v });
    },
    [controlNetId, processorChanged]
  );

  const handleDetectResolutionReset = useCallback(() => {
    processorChanged(controlNetId, {
      detect_resolution: DEFAULTS.detect_resolution,
    });
  }, [controlNetId, processorChanged]);

  const handleImageResolutionReset = useCallback(() => {
    processorChanged(controlNetId, {
      image_resolution: DEFAULTS.image_resolution,
    });
  }, [controlNetId, processorChanged]);

  const handleThrDReset = useCallback(() => {
    processorChanged(controlNetId, { thr_d: DEFAULTS.thr_d });
  }, [controlNetId, processorChanged]);

  const handleThrVReset = useCallback(() => {
    processorChanged(controlNetId, { thr_v: DEFAULTS.thr_v });
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
        value={thr_d}
        onChange={handleThrDChanged}
        handleReset={handleThrDReset}
        withReset
        min={0}
        max={1}
        step={0.01}
        withInput
        withSliderMarks
        isDisabled={!isEnabled}
      />
      <IAISlider
        label={t('controlnet.h')}
        value={thr_v}
        onChange={handleThrVChanged}
        handleReset={handleThrVReset}
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

export default memo(MlsdImageProcessor);
