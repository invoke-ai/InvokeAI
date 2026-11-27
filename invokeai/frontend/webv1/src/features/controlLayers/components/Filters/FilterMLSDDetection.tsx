import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import type { MLSDDetectionFilterConfig } from 'features/controlLayers/store/filters';
import { IMAGE_FILTERS } from 'features/controlLayers/store/filters';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import type { FilterComponentProps } from './types';

type Props = FilterComponentProps<MLSDDetectionFilterConfig>;
const DEFAULTS = IMAGE_FILTERS.mlsd_detection.buildDefaults();

export const FilterMLSDDetection = memo(({ onChange, config }: Props) => {
  const { t } = useTranslation();

  const onDistanceThresholdChanged = useCallback(
    (v: number) => {
      onChange({ ...config, distance_threshold: v });
    },
    [config, onChange]
  );

  const onScoreThresholdChanged = useCallback(
    (v: number) => {
      onChange({ ...config, score_threshold: v });
    },
    [config, onChange]
  );

  return (
    <>
      <FormControl>
        <FormLabel m={0}>{t('controlLayers.filter.mlsd_detection.score_threshold')} </FormLabel>
        <CompositeSlider
          value={config.score_threshold}
          onChange={onScoreThresholdChanged}
          defaultValue={DEFAULTS.score_threshold}
          min={0}
          max={1}
          step={0.01}
          marks
        />
        <CompositeNumberInput
          value={config.score_threshold}
          onChange={onScoreThresholdChanged}
          defaultValue={DEFAULTS.score_threshold}
          min={0}
          max={1}
          step={0.01}
        />
      </FormControl>
      <FormControl>
        <FormLabel m={0}>{t('controlLayers.filter.mlsd_detection.distance_threshold')} </FormLabel>
        <CompositeSlider
          value={config.distance_threshold}
          onChange={onDistanceThresholdChanged}
          defaultValue={DEFAULTS.distance_threshold}
          min={0}
          max={100}
          step={1}
          marks
        />
        <CompositeNumberInput
          value={config.distance_threshold}
          onChange={onDistanceThresholdChanged}
          defaultValue={DEFAULTS.distance_threshold}
          min={0}
          max={1000}
          step={1}
        />
      </FormControl>
    </>
  );
});

FilterMLSDDetection.displayName = 'FilterMLSDDetection';
