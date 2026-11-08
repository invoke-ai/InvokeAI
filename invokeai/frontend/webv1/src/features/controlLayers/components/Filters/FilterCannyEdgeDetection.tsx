import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import type { CannyEdgeDetectionFilterConfig } from 'features/controlLayers/store/filters';
import { IMAGE_FILTERS } from 'features/controlLayers/store/filters';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import type { FilterComponentProps } from './types';

type Props = FilterComponentProps<CannyEdgeDetectionFilterConfig>;
const DEFAULTS = IMAGE_FILTERS.canny_edge_detection.buildDefaults();

export const FilterCannyEdgeDetection = ({ onChange, config }: Props) => {
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
    <>
      <FormControl>
        <FormLabel m={0}>{t('controlLayers.filter.canny_edge_detection.low_threshold')}</FormLabel>
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
        <FormLabel m={0}>{t('controlLayers.filter.canny_edge_detection.high_threshold')}</FormLabel>
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
    </>
  );
};

FilterCannyEdgeDetection.displayName = 'FilterCannyEdgeDetection';
