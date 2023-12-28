import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { useProcessorNodeChanged } from 'features/controlAdapters/components/hooks/useProcessorNodeChanged';
import { CONTROLNET_PROCESSORS } from 'features/controlAdapters/store/constants';
import type { RequiredCannyImageProcessorInvocation } from 'features/controlAdapters/store/types';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import ProcessorWrapper from './common/ProcessorWrapper';

const DEFAULTS = CONTROLNET_PROCESSORS.canny_image_processor
  .default as RequiredCannyImageProcessorInvocation;

type CannyProcessorProps = {
  controlNetId: string;
  processorNode: RequiredCannyImageProcessorInvocation;
  isEnabled: boolean;
};

const CannyProcessor = (props: CannyProcessorProps) => {
  const { controlNetId, processorNode, isEnabled } = props;
  const { low_threshold, high_threshold } = processorNode;
  const processorChanged = useProcessorNodeChanged();
  const { t } = useTranslation();

  const handleLowThresholdChanged = useCallback(
    (v: number) => {
      processorChanged(controlNetId, { low_threshold: v });
    },
    [controlNetId, processorChanged]
  );

  const handleLowThresholdReset = useCallback(() => {
    processorChanged(controlNetId, {
      low_threshold: DEFAULTS.low_threshold,
    });
  }, [controlNetId, processorChanged]);

  const handleHighThresholdChanged = useCallback(
    (v: number) => {
      processorChanged(controlNetId, { high_threshold: v });
    },
    [controlNetId, processorChanged]
  );

  const handleHighThresholdReset = useCallback(() => {
    processorChanged(controlNetId, {
      high_threshold: DEFAULTS.high_threshold,
    });
  }, [controlNetId, processorChanged]);

  return (
    <ProcessorWrapper>
      <InvControl label={t('controlnet.lowThreshold')} isDisabled={!isEnabled}>
        <InvSlider
          value={low_threshold}
          onChange={handleLowThresholdChanged}
          onReset={handleLowThresholdReset}
          min={0}
          max={255}
          withNumberInput
        />
      </InvControl>
      <InvControl label={t('controlnet.highThreshold')} isDisabled={!isEnabled}>
        <InvSlider
          value={high_threshold}
          onChange={handleHighThresholdChanged}
          onReset={handleHighThresholdReset}
          min={0}
          max={255}
          withNumberInput
        />
      </InvControl>
    </ProcessorWrapper>
  );
};

export default memo(CannyProcessor);
