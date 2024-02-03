import { FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { setSeamlessXAxis } from 'features/parameters/store/generationSlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamSeamlessXAxis = () => {
  const { t } = useTranslation();
  const seamlessXAxis = useAppSelector((s) => s.generation.seamlessXAxis);

  const dispatch = useAppDispatch();

  const handleChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(setSeamlessXAxis(e.target.checked));
    },
    [dispatch]
  );

  return (
    <FormControl>
      <FormLabel>{t('parameters.seamlessXAxis')}</FormLabel>
      <Switch isChecked={seamlessXAxis} onChange={handleChange} />
    </FormControl>
  );
};

export default memo(ParamSeamlessXAxis);
