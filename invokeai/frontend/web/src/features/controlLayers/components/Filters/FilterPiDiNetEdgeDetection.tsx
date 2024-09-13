import { FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import type { PiDiNetEdgeDetectionFilterConfig } from 'features/controlLayers/store/filters';
import type { ChangeEvent } from 'react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import type { FilterComponentProps } from './types';

type Props = FilterComponentProps<PiDiNetEdgeDetectionFilterConfig>;

export const FilterPiDiNetEdgeDetection = ({ onChange, config }: Props) => {
  const { t } = useTranslation();

  const onScribbleChanged = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      onChange({ ...config, scribble: e.target.checked });
    },
    [config, onChange]
  );

  const onQuantizeEdgesChanged = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      onChange({ ...config, quantize_edges: e.target.checked });
    },
    [config, onChange]
  );

  return (
    <>
      <FormControl>
        <FormLabel m={0}>{t('controlLayers.filter.pidi_edge_detection.scribble')}</FormLabel>
        <Switch isChecked={config.scribble} onChange={onScribbleChanged} />
      </FormControl>
      <FormControl>
        <FormLabel m={0}>{t('controlLayers.filter.pidi_edge_detection.quantize_edges')}</FormLabel>
        <Switch isChecked={config.quantize_edges} onChange={onQuantizeEdgesChanged} />
      </FormControl>
    </>
  );
};

FilterPiDiNetEdgeDetection.displayName = 'FilterPiDiNetEdgeDetection';
