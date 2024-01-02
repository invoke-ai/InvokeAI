import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { useProcessorNodeChanged } from 'features/controlAdapters/components/hooks/useProcessorNodeChanged';
import { CONTROLNET_PROCESSORS } from 'features/controlAdapters/store/constants';
import type { RequiredMidasDepthImageProcessorInvocation } from 'features/controlAdapters/store/types';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import ProcessorWrapper from './common/ProcessorWrapper';

const DEFAULTS = CONTROLNET_PROCESSORS.midas_depth_image_processor
  .default as RequiredMidasDepthImageProcessorInvocation;

type Props = {
  controlNetId: string;
  processorNode: RequiredMidasDepthImageProcessorInvocation;
  isEnabled: boolean;
};

const MidasDepthProcessor = (props: Props) => {
  const { controlNetId, processorNode, isEnabled } = props;
  const { a_mult, bg_th } = processorNode;
  const processorChanged = useProcessorNodeChanged();
  const { t } = useTranslation();

  const handleAMultChanged = useCallback(
    (v: number) => {
      processorChanged(controlNetId, { a_mult: v });
    },
    [controlNetId, processorChanged]
  );

  const handleBgThChanged = useCallback(
    (v: number) => {
      processorChanged(controlNetId, { bg_th: v });
    },
    [controlNetId, processorChanged]
  );

  const handleAMultReset = useCallback(() => {
    processorChanged(controlNetId, { a_mult: DEFAULTS.a_mult });
  }, [controlNetId, processorChanged]);

  const handleBgThReset = useCallback(() => {
    processorChanged(controlNetId, { bg_th: DEFAULTS.bg_th });
  }, [controlNetId, processorChanged]);

  return (
    <ProcessorWrapper>
      <InvControl label={t('controlnet.amult')} isDisabled={!isEnabled}>
        <InvSlider
          value={a_mult}
          onChange={handleAMultChanged}
          onReset={handleAMultReset}
          min={0}
          max={20}
          step={0.01}
          marks
          withNumberInput
        />
      </InvControl>
      <InvControl label={t('controlnet.bgth')} isDisabled={!isEnabled}>
        <InvSlider
          value={bg_th}
          onChange={handleBgThChanged}
          onReset={handleBgThReset}
          min={0}
          max={20}
          step={0.01}
          marks
          withNumberInput
        />
      </InvControl>
    </ProcessorWrapper>
  );
};

export default memo(MidasDepthProcessor);
