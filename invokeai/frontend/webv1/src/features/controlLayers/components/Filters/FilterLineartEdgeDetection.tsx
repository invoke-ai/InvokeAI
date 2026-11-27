import { FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import type { LineartEdgeDetectionFilterConfig } from 'features/controlLayers/store/filters';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import type { FilterComponentProps } from './types';

type Props = FilterComponentProps<LineartEdgeDetectionFilterConfig>;

export const FilterLineartEdgeDetection = memo(({ onChange, config }: Props) => {
  const { t } = useTranslation();

  const handleCoarseChanged = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      onChange({ ...config, coarse: e.target.checked });
    },
    [config, onChange]
  );

  return (
    <>
      <FormControl>
        <FormLabel m={0}>{t('controlLayers.filter.lineart_edge_detection.coarse')}</FormLabel>
        <Switch isChecked={config.coarse} onChange={handleCoarseChanged} />
      </FormControl>
    </>
  );
});

FilterLineartEdgeDetection.displayName = 'FilterLineartEdgeDetection';
