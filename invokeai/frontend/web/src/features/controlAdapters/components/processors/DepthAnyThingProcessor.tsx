import type { ComboboxOnChange } from '@invoke-ai/ui-library';
import { Combobox, CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useProcessorNodeChanged } from 'features/controlAdapters/components/hooks/useProcessorNodeChanged';
import { useGetDefaultForControlnetProcessor } from 'features/controlAdapters/hooks/useGetDefaultForControlnetProcessor';
import type {
  DepthAnythingModelSize,
  RequiredDepthAnythingImageProcessorInvocation,
} from 'features/controlAdapters/store/types';
import { isDepthAnythingModelSize } from 'features/controlAdapters/store/types';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import ProcessorWrapper from './common/ProcessorWrapper';

type Props = {
  controlNetId: string;
  processorNode: RequiredDepthAnythingImageProcessorInvocation;
  isEnabled: boolean;
};

const DepthAnythingProcessor = (props: Props) => {
  const { controlNetId, processorNode, isEnabled } = props;
  const { model_size, resolution } = processorNode;
  const processorChanged = useProcessorNodeChanged();
  const { t } = useTranslation();

  const defaults = useGetDefaultForControlnetProcessor(
    'midas_depth_image_processor'
  ) as RequiredDepthAnythingImageProcessorInvocation;

  const handleModelSizeChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!isDepthAnythingModelSize(v?.value)) {
        return;
      }
      processorChanged(controlNetId, {
        model_size: v.value,
      });
    },
    [controlNetId, processorChanged]
  );

  const options: { label: string; value: DepthAnythingModelSize }[] = useMemo(
    () => [
      { label: t('controlnet.small'), value: 'small' },
      { label: t('controlnet.base'), value: 'base' },
      { label: t('controlnet.large'), value: 'large' },
    ],
    [t]
  );

  const value = useMemo(() => options.filter((o) => o.value === model_size)[0], [options, model_size]);

  const handleResolutionChange = useCallback(
    (v: number) => {
      processorChanged(controlNetId, { resolution: v });
    },
    [controlNetId, processorChanged]
  );

  const handleResolutionDefaultChange = useCallback(() => {
    processorChanged(controlNetId, { resolution: 512 });
  }, [controlNetId, processorChanged]);

  return (
    <ProcessorWrapper>
      <FormControl isDisabled={!isEnabled}>
        <FormLabel>{t('controlnet.modelSize')}</FormLabel>
        <Combobox
          value={value}
          defaultInputValue={defaults.model_size}
          options={options}
          onChange={handleModelSizeChange}
        />
      </FormControl>
      <FormControl isDisabled={!isEnabled}>
        <FormLabel>{t('controlnet.imageResolution')}</FormLabel>
        <CompositeSlider
          value={resolution}
          onChange={handleResolutionChange}
          defaultValue={defaults.resolution}
          min={64}
          max={4096}
          step={64}
          marks
          onReset={handleResolutionDefaultChange}
        />
        <CompositeNumberInput
          value={resolution}
          onChange={handleResolutionChange}
          defaultValue={defaults.resolution}
          min={64}
          max={4096}
          step={64}
        />
      </FormControl>
    </ProcessorWrapper>
  );
};

export default memo(DepthAnythingProcessor);
