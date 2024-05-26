import { FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import type { ProcessorComponentProps } from 'features/controlLayers/components/ControlAndIPAdapter/processors/types';
import type { HedProcessorConfig } from 'features/controlLayers/util/controlAdapters';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import ProcessorWrapper from './ProcessorWrapper';

type Props = ProcessorComponentProps<HedProcessorConfig>;

export const HedProcessor = memo(({ onChange, config }: Props) => {
  const { t } = useTranslation();

  const handleScribbleChanged = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      onChange({ ...config, scribble: e.target.checked });
    },
    [config, onChange]
  );

  return (
    <ProcessorWrapper>
      <FormControl>
        <FormLabel m={0}>{t('controlnet.scribble')}</FormLabel>
        <Switch isChecked={config.scribble} onChange={handleScribbleChanged} />
      </FormControl>
    </ProcessorWrapper>
  );
});

HedProcessor.displayName = 'HedProcessor';
