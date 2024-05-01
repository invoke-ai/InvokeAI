import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import type { ProcessorComponentProps } from 'features/controlLayers/components/ControlAndIPAdapter/processors/types';
import type { MlsdProcessorConfig } from 'features/controlLayers/util/controlAdapters';
import { CONTROLNET_PROCESSORS } from 'features/controlLayers/util/controlAdapters';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import ProcessorWrapper from './ProcessorWrapper';

type Props = ProcessorComponentProps<MlsdProcessorConfig>;
const DEFAULTS = CONTROLNET_PROCESSORS['mlsd_image_processor'].buildDefaults();

export const MlsdImageProcessor = memo(({ onChange, config }: Props) => {
  const { t } = useTranslation();

  const handleThrDChanged = useCallback(
    (v: number) => {
      onChange({ ...config, thr_d: v });
    },
    [config, onChange]
  );

  const handleThrVChanged = useCallback(
    (v: number) => {
      onChange({ ...config, thr_v: v });
    },
    [config, onChange]
  );

  return (
    <ProcessorWrapper>
      <FormControl>
        <FormLabel m={0}>{t('controlnet.w')} </FormLabel>
        <CompositeSlider
          value={config.thr_d}
          onChange={handleThrDChanged}
          defaultValue={DEFAULTS.thr_d}
          min={0}
          max={1}
          step={0.01}
          marks
        />
        <CompositeNumberInput
          value={config.thr_d}
          onChange={handleThrDChanged}
          defaultValue={DEFAULTS.thr_d}
          min={0}
          max={1}
          step={0.01}
        />
      </FormControl>
      <FormControl>
        <FormLabel m={0}>{t('controlnet.h')} </FormLabel>
        <CompositeSlider
          value={config.thr_v}
          onChange={handleThrVChanged}
          defaultValue={DEFAULTS.thr_v}
          min={0}
          max={1}
          step={0.01}
          marks
        />
        <CompositeNumberInput
          value={config.thr_v}
          onChange={handleThrVChanged}
          defaultValue={DEFAULTS.thr_v}
          min={0}
          max={1}
          step={0.01}
        />
      </FormControl>
    </ProcessorWrapper>
  );
});

MlsdImageProcessor.displayName = 'MlsdImageProcessor';
