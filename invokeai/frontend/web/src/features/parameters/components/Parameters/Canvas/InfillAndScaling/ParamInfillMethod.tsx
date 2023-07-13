import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIMantineSelect from 'common/components/IAIMantineSelect';
import { setInfillMethod } from 'features/parameters/store/generationSlice';

import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetAppConfigQuery } from 'services/api/endpoints/appInfo';

const selector = createSelector(
  [stateSelector],
  ({ generation }) => {
    const { infillMethod } = generation;

    return {
      infillMethod,
    };
  },
  defaultSelectorOptions
);

const ParamInfillMethod = () => {
  const dispatch = useAppDispatch();
  const { infillMethod } = useAppSelector(selector);

  const { data: appConfigData, isLoading } = useGetAppConfigQuery();

  const infill_methods = appConfigData?.infill_methods;

  const { t } = useTranslation();

  const handleChange = useCallback(
    (v: string) => {
      dispatch(setInfillMethod(v));
    },
    [dispatch]
  );

  return (
    <IAIMantineSelect
      disabled={infill_methods?.length === 0}
      placeholder={isLoading ? 'Loading...' : undefined}
      label={t('parameters.infillMethod')}
      value={infillMethod}
      data={infill_methods ?? []}
      onChange={handleChange}
    />
  );
};

export default memo(ParamInfillMethod);
