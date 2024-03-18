import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useProcessorNodeChanged } from 'features/controlAdapters/components/hooks/useProcessorNodeChanged';
import { useGetDefaultForControlnetProcessor } from 'features/controlAdapters/hooks/useGetDefaultForControlnetProcessor';
import type { RequiredMidasDepthImageProcessorInvocation } from 'features/controlAdapters/store/types';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import ProcessorWrapper from './common/ProcessorWrapper';

type Props = {
  controlNetId: string;
  processorNode: RequiredMidasDepthImageProcessorInvocation;
  isEnabled: boolean;
};

const MidasDepthProcessor = (props: Props) => {
  const { controlNetId, processorNode, isEnabled } = props;
  const { a_mult, bg_th, image_resolution } = processorNode;
  const processorChanged = useProcessorNodeChanged();
  const { t } = useTranslation();

  const defaults = useGetDefaultForControlnetProcessor(
    'midas_depth_image_processor'
  ) as RequiredMidasDepthImageProcessorInvocation;

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

  const handleImageResolutionChanged = useCallback(
    (v: number) => {
      processorChanged(controlNetId, { image_resolution: v });
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
          defaultValue={defaults.a_mult}
          min={0}
          max={20}
          step={0.01}
          marks
        />
        <CompositeNumberInput
          value={a_mult}
          onChange={handleAMultChanged}
          defaultValue={defaults.a_mult}
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
          defaultValue={defaults.bg_th}
          min={0}
          max={20}
          step={0.01}
          marks
        />
        <CompositeNumberInput
          value={bg_th}
          onChange={handleBgThChanged}
          defaultValue={defaults.bg_th}
          min={0}
          max={20}
          step={0.01}
        />
      </FormControl>
      <FormControl isDisabled={!isEnabled}>
        <FormLabel>{t('controlnet.imageResolution')}</FormLabel>
        <CompositeSlider
          value={image_resolution}
          onChange={handleImageResolutionChanged}
          defaultValue={defaults.image_resolution}
          min={0}
          max={4096}
          marks
        />
        <CompositeNumberInput
          value={image_resolution}
          onChange={handleImageResolutionChanged}
          defaultValue={defaults.image_resolution}
          min={0}
          max={4096}
        />
      </FormControl>
    </ProcessorWrapper>
  );
};

export default memo(MidasDepthProcessor);
