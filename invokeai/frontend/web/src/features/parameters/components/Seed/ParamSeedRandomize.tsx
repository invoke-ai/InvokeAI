import { FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import {
  selectShouldRandomizeSeed,
  setShouldRandomizeSeed,
  useParamsDispatch,
} from 'features/controlLayers/store/paramsSlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const ParamSeedRandomize = memo(() => {
  const dispatchParams = useParamsDispatch();
  const { t } = useTranslation();

  const shouldRandomizeSeed = useAppSelector(selectShouldRandomizeSeed);

  const handleChangeShouldRandomizeSeed = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => dispatchParams(setShouldRandomizeSeed, e.target.checked),
    [dispatchParams]
  );

  return (
    <FormControl w="min-content">
      <FormLabel m={0}>{t('common.random')}</FormLabel>
      <Switch isChecked={shouldRandomizeSeed} onChange={handleChangeShouldRandomizeSeed} />
    </FormControl>
  );
});

ParamSeedRandomize.displayName = 'ParamSeedRandomize';
