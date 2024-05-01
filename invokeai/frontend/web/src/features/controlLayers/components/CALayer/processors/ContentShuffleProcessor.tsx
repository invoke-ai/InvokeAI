import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import type { ProcessorComponentProps } from 'features/controlLayers/components/CALayer/processors/types';
import type { ContentShuffleProcessorConfig } from 'features/controlLayers/util/controlAdapters';
import { CONTROLNET_PROCESSORS } from 'features/controlLayers/util/controlAdapters';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import ProcessorWrapper from './ProcessorWrapper';

type Props = ProcessorComponentProps<ContentShuffleProcessorConfig>;
const DEFAULTS = CONTROLNET_PROCESSORS['content_shuffle_image_processor'].buildDefaults();

export const ContentShuffleProcessor = memo(({ onChange, config }: Props) => {
  const { t } = useTranslation();

  const handleWChanged = useCallback(
    (v: number) => {
      onChange({ ...config, w: v });
    },
    [config, onChange]
  );

  const handleHChanged = useCallback(
    (v: number) => {
      onChange({ ...config, h: v });
    },
    [config, onChange]
  );

  const handleFChanged = useCallback(
    (v: number) => {
      onChange({ ...config, f: v });
    },
    [config, onChange]
  );

  return (
    <ProcessorWrapper>
      <FormControl>
        <FormLabel>{t('controlnet.w')}</FormLabel>
        <CompositeSlider
          value={config.w}
          defaultValue={DEFAULTS.w}
          onChange={handleWChanged}
          min={0}
          max={4096}
          marks
        />
        <CompositeNumberInput value={config.w} defaultValue={DEFAULTS.w} onChange={handleWChanged} min={0} max={4096} />
      </FormControl>
      <FormControl>
        <FormLabel>{t('controlnet.h')}</FormLabel>
        <CompositeSlider
          value={config.h}
          defaultValue={DEFAULTS.h}
          onChange={handleHChanged}
          min={0}
          max={4096}
          marks
        />
        <CompositeNumberInput value={config.h} defaultValue={DEFAULTS.h} onChange={handleHChanged} min={0} max={4096} />
      </FormControl>
      <FormControl>
        <FormLabel>{t('controlnet.f')}</FormLabel>
        <CompositeSlider
          value={config.f}
          defaultValue={DEFAULTS.f}
          onChange={handleFChanged}
          min={0}
          max={4096}
          marks
        />
        <CompositeNumberInput value={config.f} defaultValue={DEFAULTS.f} onChange={handleFChanged} min={0} max={4096} />
      </FormControl>
    </ProcessorWrapper>
  );
});

ContentShuffleProcessor.displayName = 'ContentShuffleProcessor';
