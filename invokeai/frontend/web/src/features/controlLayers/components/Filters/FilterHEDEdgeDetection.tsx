import { FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import type { HEDEdgeDetectionFilterConfig } from 'features/controlLayers/store/filters';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import type { FilterComponentProps } from './types';

type Props = FilterComponentProps<HEDEdgeDetectionFilterConfig>;

export const FilterHEDEdgeDetection = memo(({ onChange, config }: Props) => {
  const { t } = useTranslation();

  const handleScribbleChanged = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      onChange({ ...config, scribble: e.target.checked });
    },
    [config, onChange]
  );

  return (
    <>
      <FormControl>
        <FormLabel m={0}>{t('controlLayers.filter.hed_edge_detection.scribble')}</FormLabel>
        <Switch isChecked={config.scribble} onChange={handleScribbleChanged} />
      </FormControl>
    </>
  );
});

FilterHEDEdgeDetection.displayName = 'FilterHEDEdgeDetection';
