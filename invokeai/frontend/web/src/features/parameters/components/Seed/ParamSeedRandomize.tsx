import { FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { setShouldRandomizeSeed } from 'features/controlLayers/store/canvasV2Slice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const ParamSeedRandomize = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const shouldRandomizeSeed = useAppSelector((s) => s.canvasV2.params.shouldRandomizeSeed);

  const handleChangeShouldRandomizeSeed = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => dispatch(setShouldRandomizeSeed(e.target.checked)),
    [dispatch]
  );

  return (
    <FormControl w="min-content">
      <FormLabel>{t('common.random')}</FormLabel>
      <Switch isChecked={shouldRandomizeSeed} onChange={handleChangeShouldRandomizeSeed} />
    </FormControl>
  );
});

ParamSeedRandomize.displayName = 'ParamSeedRandomize';
