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
      <InvControl
        label={t('controlnet.detectResolution')}
        isDisabled={!isEnabled}
      >
        <InvSlider
          value={detect_resolution}
          defaultValue={DEFAULTS.detect_resolution}
          onChange={handleDetectResolutionChanged}
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
          defaultValue={DEFAULTS.image_resolution}
          onChange={handleImageResolutionChanged}
          min={0}
          max={4096}
          marks
          withNumberInput
        />
      </InvControl>
      <InvControl label={t('controlnet.w')} isDisabled={!isEnabled}>
        <InvSlider
          value={w}
          defaultValue={DEFAULTS.w}
          onChange={handleWChanged}
          min={0}
          max={4096}
          marks
          withNumberInput
        />
      </InvControl>
      <InvControl label={t('controlnet.h')} isDisabled={!isEnabled}>
        <InvSlider
          value={h}
          defaultValue={DEFAULTS.h}
          onChange={handleHChanged}
          min={0}
          max={4096}
          marks
          withNumberInput
        />
      </InvControl>
      <InvControl label={t('controlnet.f')} isDisabled={!isEnabled}>
        <InvSlider
          value={f}
          defaultValue={DEFAULTS.f}
          onChange={handleFChanged}
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
