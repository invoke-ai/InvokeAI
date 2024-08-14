import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import type { ContentShuffleProcessorConfig } from 'features/controlLayers/store/types';
import { IMAGE_FILTERS } from 'features/controlLayers/store/types';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import type { FilterComponentProps } from './types';

type Props = FilterComponentProps<ContentShuffleProcessorConfig>;
const DEFAULTS = IMAGE_FILTERS['content_shuffle_image_processor'].buildDefaults();

export const FilterContentShuffle = memo(({ onChange, config }: Props) => {
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
    <>
      <FormControl>
        <FormLabel m={0}>{t('controlnet.w')}</FormLabel>
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
        <FormLabel m={0}>{t('controlnet.h')}</FormLabel>
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
        <FormLabel m={0}>{t('controlnet.f')}</FormLabel>
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
    </>
  );
});

FilterContentShuffle.displayName = 'FilterContentShuffle';
