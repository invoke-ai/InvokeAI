import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useProcessorNodeChanged } from 'features/controlAdapters/components/hooks/useProcessorNodeChanged';
import { useGetDefaultForControlnetProcessor } from 'features/controlAdapters/hooks/useGetDefaultForControlnetProcessor';
import type { RequiredMlsdImageProcessorInvocation } from 'features/controlAdapters/store/types';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import ProcessorWrapper from './common/ProcessorWrapper';

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

  const defaults = useGetDefaultForControlnetProcessor('mlsd_image_processor') as RequiredMlsdImageProcessorInvocation;

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

  return (
    <ProcessorWrapper>
      <FormControl isDisabled={!isEnabled}>
        <FormLabel>{t('controlnet.detectResolution')}</FormLabel>
        <CompositeSlider
          value={detect_resolution}
          onChange={handleDetectResolutionChanged}
          defaultValue={defaults.detect_resolution}
          min={0}
          max={4096}
          marks={marks0to4096}
        />
        <CompositeNumberInput
          value={detect_resolution}
          onChange={handleDetectResolutionChanged}
          defaultValue={defaults.detect_resolution}
          min={0}
          max={4096}
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
          marks={marks0to4096}
        />
        <CompositeNumberInput
          value={image_resolution}
          onChange={handleImageResolutionChanged}
          defaultValue={defaults.image_resolution}
          min={0}
          max={4096}
        />
      </FormControl>
      <FormControl isDisabled={!isEnabled}>
        <FormLabel>{t('controlnet.w')} </FormLabel>
        <CompositeSlider
          value={thr_d}
          onChange={handleThrDChanged}
          defaultValue={defaults.thr_d}
          min={0}
          max={1}
          step={0.01}
          marks={marks0to1}
        />
        <CompositeNumberInput
          value={thr_d}
          onChange={handleThrDChanged}
          defaultValue={defaults.thr_d}
          min={0}
          max={1}
          step={0.01}
        />
      </FormControl>
      <FormControl isDisabled={!isEnabled}>
        <FormLabel>{t('controlnet.h')} </FormLabel>
        <CompositeSlider
          value={thr_v}
          onChange={handleThrVChanged}
          defaultValue={defaults.thr_v}
          min={0}
          max={1}
          step={0.01}
          marks={marks0to1}
        />
        <CompositeNumberInput
          value={thr_v}
          onChange={handleThrVChanged}
          defaultValue={defaults.thr_v}
          min={0}
          max={1}
          step={0.01}
        />
      </FormControl>
    </ProcessorWrapper>
  );
};

export default memo(MlsdImageProcessor);

const marks0to4096 = [0, 4096];
const marks0to1 = [0, 1];
