import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { useProcessorNodeChanged } from 'features/controlAdapters/components/hooks/useProcessorNodeChanged';
import { CONTROLNET_PROCESSORS } from 'features/controlAdapters/store/constants';
import type { RequiredContentShuffleImageProcessorInvocation } from 'features/controlAdapters/store/types';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

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
          marks
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
          marks
          withNumberInput
        />
      </InvControl>
      <InvControl label={t('controlnet.w')} isDisabled={!isEnabled}>
        <InvSlider
          value={w}
          onChange={handleWChanged}
          onReset={handleWReset}
          min={0}
          max={4096}
          marks
          withNumberInput
        />
      </InvControl>
      <InvControl label={t('controlnet.h')} isDisabled={!isEnabled}>
        <InvSlider
          value={h}
          onChange={handleHChanged}
          onReset={handleHReset}
          min={0}
          max={4096}
          marks
          withNumberInput
        />
      </InvControl>
      <InvControl label={t('controlnet.f')} isDisabled={!isEnabled}>
        <InvSlider
          value={f}
          onChange={handleFChanged}
          onReset={handleFReset}
          min={0}
          max={4096}
          marks
          withNumberInput
        />
      </InvControl>
    </ProcessorWrapper>
  );
};

export default memo(ContentShuffleProcessor);
