import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import type { MlsdProcessorConfig } from 'features/controlLayers/store/types';
import { IMAGE_FILTERS } from 'features/controlLayers/store/types';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import type { FilterComponentProps } from './types';

type Props = FilterComponentProps<MlsdProcessorConfig>;
const DEFAULTS = IMAGE_FILTERS['mlsd_image_processor'].buildDefaults();

export const FilterMlsdImage = memo(({ onChange, config }: Props) => {
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
    <>
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
    </>
  );
});

FilterMlsdImage.displayName = 'FilterMlsdImage';
