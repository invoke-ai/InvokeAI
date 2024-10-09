import { FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import {
  selectOptimizedDenoisingEnabled,
  setOptimizedDenoisingEnabled,
} from 'features/controlLayers/store/paramsSlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const ParamOptimizedDenoisingToggle = memo(() => {
  const optimizedDenoisingEnabled = useAppSelector(selectOptimizedDenoisingEnabled);
  const dispatch = useAppDispatch();

  const onChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      dispatch(setOptimizedDenoisingEnabled(event.target.checked));
    },
    [dispatch]
  );

  const { t } = useTranslation();

  return (
    <FormControl w="min-content">
      <InformationalPopover feature="optimizedDenoising">
        <FormLabel m={0}>
          {t('parameters.optimizedImageToImage')} ({t('settings.beta')})
        </FormLabel>
      </InformationalPopover>
      <Switch isChecked={optimizedDenoisingEnabled} onChange={onChange} />
    </FormControl>
  );
});

ParamOptimizedDenoisingToggle.displayName = 'ParamOptimizedDenoisingToggle';
