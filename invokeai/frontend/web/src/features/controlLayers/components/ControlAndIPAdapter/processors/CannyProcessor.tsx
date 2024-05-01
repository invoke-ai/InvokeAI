import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import type { ProcessorComponentProps } from 'features/controlLayers/components/ControlAndIPAdapter/processors/types';
import { type CannyProcessorConfig, CONTROLNET_PROCESSORS } from 'features/controlLayers/util/controlAdapters';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import ProcessorWrapper from './ProcessorWrapper';

type Props = ProcessorComponentProps<CannyProcessorConfig>;
const DEFAULTS = CONTROLNET_PROCESSORS['canny_image_processor'].buildDefaults();

export const CannyProcessor = ({ onChange, config }: Props) => {
  const { t } = useTranslation();
  const handleLowThresholdChanged = useCallback(
    (v: number) => {
      onChange({ ...config, low_threshold: v });
    },
    [onChange, config]
  );
  const handleHighThresholdChanged = useCallback(
    (v: number) => {
      onChange({ ...config, high_threshold: v });
    },
    [onChange, config]
  );

  return (
    <ProcessorWrapper>
      <FormControl>
        <FormLabel m={0}>{t('controlnet.lowThreshold')}</FormLabel>
        <CompositeSlider
          value={config.low_threshold}
          onChange={handleLowThresholdChanged}
          defaultValue={DEFAULTS.low_threshold}
          min={0}
          max={255}
        />
        <CompositeNumberInput
          value={config.low_threshold}
          onChange={handleLowThresholdChanged}
          defaultValue={DEFAULTS.low_threshold}
          min={0}
          max={255}
        />
      </FormControl>
      <FormControl>
        <FormLabel m={0}>{t('controlnet.highThreshold')}</FormLabel>
        <CompositeSlider
          value={config.high_threshold}
          onChange={handleHighThresholdChanged}
          defaultValue={DEFAULTS.high_threshold}
          min={0}
          max={255}
        />
        <CompositeNumberInput
          value={config.high_threshold}
          onChange={handleHighThresholdChanged}
          defaultValue={DEFAULTS.high_threshold}
          min={0}
          max={255}
        />
      </FormControl>
    </ProcessorWrapper>
  );
};

CannyProcessor.displayName = 'CannyProcessor';
