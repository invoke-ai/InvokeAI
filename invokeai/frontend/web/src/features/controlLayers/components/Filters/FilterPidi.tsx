import { FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import type { PidiProcessorConfig } from 'features/controlLayers/store/types';
import type { ChangeEvent } from 'react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import type { FilterComponentProps } from './types';

type Props = FilterComponentProps<PidiProcessorConfig>;

export const FilterPidi = ({ onChange, config }: Props) => {
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
    <>
      <FormControl>
        <FormLabel m={0}>{t('controlnet.scribble')}</FormLabel>
        <Switch isChecked={config.scribble} onChange={handleScribbleChanged} />
      </FormControl>
      <FormControl>
        <FormLabel m={0}>{t('controlnet.safe')}</FormLabel>
        <Switch isChecked={config.safe} onChange={handleSafeChanged} />
      </FormControl>
    </>
  );
};

FilterPidi.displayName = 'FilterPidi';
