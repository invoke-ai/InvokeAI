import { FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
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
      <InformationalPopover feature="seamlessTilingYAxis">
        <FormLabel>{t('parameters.seamlessYAxis')}</FormLabel>
      </InformationalPopover>
      <Switch isChecked={seamlessYAxis} onChange={handleChange} />
    </FormControl>
  );
};

export default memo(ParamSeamlessYAxis);
