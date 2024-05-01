import { FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import type { ProcessorComponentProps } from 'features/controlLayers/components/ControlAndIPAdapter/processors/types';
import type { PidiProcessorConfig } from 'features/controlLayers/util/controlAdapters';
import type { ChangeEvent } from 'react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import ProcessorWrapper from './ProcessorWrapper';

type Props = ProcessorComponentProps<PidiProcessorConfig>;

export const PidiProcessor = ({ onChange, config }: Props) => {
  const { t } = useTranslation();

  const handleScribbleChanged = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      onChange({ ...config, scribble: e.target.checked });
    },
    [config, onChange]
  );

  const handleSafeChanged = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      onChange({ ...config, safe: e.target.checked });
    },
    [config, onChange]
  );

  return (
    <ProcessorWrapper>
      <FormControl>
        <FormLabel>{t('controlnet.scribble')}</FormLabel>
        <Switch isChecked={config.scribble} onChange={handleScribbleChanged} />
      </FormControl>
      <FormControl>
        <FormLabel>{t('controlnet.safe')}</FormLabel>
        <Switch isChecked={config.safe} onChange={handleSafeChanged} />
      </FormControl>
    </ProcessorWrapper>
  );
};

PidiProcessor.displayName = 'PidiProcessor';
