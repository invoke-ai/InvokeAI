import { FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { setSeamlessYAxis } from 'features/parameters/store/generationSlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamSeamlessYAxis = () => {
  const { t } = useTranslation();
  const seamlessYAxis = useAppSelector((s) => s.generation.seamlessYAxis);
  const dispatch = useAppDispatch();
  const handleChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(setSeamlessYAxis(e.target.checked));
    },
    [dispatch]
  );

  return (
    <FormControl>
      <FormLabel>{t('parameters.seamlessYAxis')}</FormLabel>
      <Switch isChecked={seamlessYAxis} onChange={handleChange} />
    </FormControl>
  );
};

export default memo(ParamSeamlessYAxis);
