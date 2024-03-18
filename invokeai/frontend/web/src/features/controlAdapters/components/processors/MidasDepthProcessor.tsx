import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
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

  return (
    <ProcessorWrapper>
      <FormControl isDisabled={!isEnabled}>
        <FormLabel>{t('controlnet.amult')}</FormLabel>
        <CompositeSlider
          value={a_mult}
          onChange={handleAMultChanged}
          defaultValue={DEFAULTS.a_mult}
          min={0}
          max={20}
          step={0.01}
          marks
        />
        <CompositeNumberInput
          value={a_mult}
          onChange={handleAMultChanged}
          defaultValue={DEFAULTS.a_mult}
          min={0}
          max={20}
          step={0.01}
        />
      </FormControl>
      <FormControl isDisabled={!isEnabled}>
        <FormLabel>{t('controlnet.bgth')}</FormLabel>
        <CompositeSlider
          value={bg_th}
          onChange={handleBgThChanged}
          defaultValue={DEFAULTS.bg_th}
          min={0}
          max={20}
          step={0.01}
          marks
        />
        <CompositeNumberInput
          value={bg_th}
          onChange={handleBgThChanged}
          defaultValue={DEFAULTS.bg_th}
          min={0}
          max={20}
          step={0.01}
        />
      </FormControl>
    </ProcessorWrapper>
  );
};

export default memo(MidasDepthProcessor);
