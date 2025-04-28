import { FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  selectImagen3EnhancePrompt,
  selectIsImagen3,
  selectShouldRandomizeSeed,
  setShouldRandomizeSeed,
} from 'features/controlLayers/store/paramsSlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const ParamSeedRandomize = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const isImagen3 = useAppSelector(selectIsImagen3);
  const imagen3EnhancePrompt = useAppSelector(selectImagen3EnhancePrompt);

  const shouldRandomizeSeed = useAppSelector(selectShouldRandomizeSeed);

  const handleChangeShouldRandomizeSeed = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => dispatch(setShouldRandomizeSeed(e.target.checked)),
    [dispatch]
  );

  return (
    <FormControl w="min-content" isDisabled={isImagen3 && imagen3EnhancePrompt}>
      <FormLabel m={0}>{t('common.random')}</FormLabel>
      <Switch isChecked={shouldRandomizeSeed} onChange={handleChangeShouldRandomizeSeed} />
    </FormControl>
  );
});

ParamSeedRandomize.displayName = 'ParamSeedRandomize';
