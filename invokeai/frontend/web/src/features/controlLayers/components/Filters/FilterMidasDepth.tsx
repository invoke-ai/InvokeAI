import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import type { MidasDepthProcessorConfig } from 'features/controlLayers/store/types';
import { IMAGE_FILTERS } from 'features/controlLayers/store/types';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import type { FilterComponentProps } from './types';

type Props = FilterComponentProps<MidasDepthProcessorConfig>;
const DEFAULTS = IMAGE_FILTERS['midas_depth_image_processor'].buildDefaults();

export const FilterMidasDepth = memo(({ onChange, config }: Props) => {
  const { t } = useTranslation();

  const handleAMultChanged = useCallback(
    (v: number) => {
      onChange({ ...config, a_mult: v });
    },
    [config, onChange]
  );

  const handleBgThChanged = useCallback(
    (v: number) => {
      onChange({ ...config, bg_th: v });
    },
    [config, onChange]
  );

  return (
    <>
      <FormControl>
        <FormLabel m={0}>{t('controlnet.amult')}</FormLabel>
        <CompositeSlider
          value={config.a_mult}
          onChange={handleAMultChanged}
          defaultValue={DEFAULTS.a_mult}
          min={0}
          max={20}
          step={0.01}
          marks
        />
        <CompositeNumberInput
          value={config.a_mult}
          onChange={handleAMultChanged}
          defaultValue={DEFAULTS.a_mult}
          min={0}
          max={20}
          step={0.01}
        />
      </FormControl>
      <FormControl>
        <FormLabel m={0}>{t('controlnet.bgth')}</FormLabel>
        <CompositeSlider
          value={config.bg_th}
          onChange={handleBgThChanged}
          defaultValue={DEFAULTS.bg_th}
          min={0}
          max={20}
          step={0.01}
          marks
        />
        <CompositeNumberInput
          value={config.bg_th}
          onChange={handleBgThChanged}
          defaultValue={DEFAULTS.bg_th}
          min={0}
          max={20}
          step={0.01}
        />
      </FormControl>
    </>
  );
});

FilterMidasDepth.displayName = 'FilterMidasDepth';
