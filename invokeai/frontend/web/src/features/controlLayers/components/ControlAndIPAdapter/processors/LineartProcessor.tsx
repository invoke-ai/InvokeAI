import { FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import type { ProcessorComponentProps } from 'features/controlLayers/components/ControlAndIPAdapter/processors/types';
import type { LineartProcessorConfig } from 'features/controlLayers/util/controlAdapters';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import ProcessorWrapper from './ProcessorWrapper';

type Props = ProcessorComponentProps<LineartProcessorConfig>;

export const LineartProcessor = memo(({ onChange, config }: Props) => {
  const { t } = useTranslation();

  const handleCoarseChanged = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      onChange({ ...config, coarse: e.target.checked });
    },
    [config, onChange]
  );

  return (
    <ProcessorWrapper>
      <FormControl>
        <FormLabel m={0}>{t('controlnet.coarse')}</FormLabel>
        <Switch isChecked={config.coarse} onChange={handleCoarseChanged} />
      </FormControl>
    </ProcessorWrapper>
  );
});

LineartProcessor.displayName = 'LineartProcessor';
