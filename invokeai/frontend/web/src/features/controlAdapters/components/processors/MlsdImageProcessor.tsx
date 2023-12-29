import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { useProcessorNodeChanged } from 'features/controlAdapters/components/hooks/useProcessorNodeChanged';
import { CONTROLNET_PROCESSORS } from 'features/controlAdapters/store/constants';
import type { RequiredMlsdImageProcessorInvocation } from 'features/controlAdapters/store/types';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

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
      <InvControl
        label={t('controlnet.detectResolution')}
        isDisabled={!isEnabled}
      >
        <InvSlider
          value={detect_resolution}
          onChange={handleDetectResolutionChanged}
          onReset={handleDetectResolutionReset}
          min={0}
          max={4096}
          marks={marks0to4096}
          withNumberInput
        />
      </InvControl>
      <InvControl
        label={t('controlnet.imageResolution')}
        isDisabled={!isEnabled}
      >
        <InvSlider
          value={image_resolution}
          onChange={handleImageResolutionChanged}
          onReset={handleImageResolutionReset}
          min={0}
          max={4096}
          marks={marks0to4096}
          withNumberInput
        />
      </InvControl>
      <InvControl label={t('controlnet.w')} isDisabled={!isEnabled}>
        <InvSlider
          value={thr_d}
          onChange={handleThrDChanged}
          onReset={handleThrDReset}
          min={0}
          max={1}
          step={0.01}
          marks={marks0to1}
          withNumberInput
        />
      </InvControl>
      <InvControl label={t('controlnet.h')} isDisabled={!isEnabled}>
        <InvSlider
          value={thr_v}
          onChange={handleThrVChanged}
          onReset={handleThrVReset}
          min={0}
          max={1}
          step={0.01}
          marks={marks0to1}
          withNumberInput
        />
      </InvControl>
    </ProcessorWrapper>
  );
};

export default memo(MlsdImageProcessor);

const marks0to4096 = [0, 4096];
const marks0to1 = [0, 1];