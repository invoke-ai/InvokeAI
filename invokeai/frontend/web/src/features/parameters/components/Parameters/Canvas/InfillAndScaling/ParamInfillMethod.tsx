import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIMantineSelect from 'common/components/IAIMantineSelect';
import { generationSelector } from 'features/parameters/store/generationSelectors';
import { setInfillMethod } from 'features/parameters/store/generationSlice';
import { systemSelector } from 'features/system/store/systemSelectors';

import { memo, useCallback, useEffect, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetAppConfigQuery } from '../../../../../../services/api/endpoints/appInfo';
import { setAvailableInfillMethods } from '../../../../../system/store/systemSlice';

const selector = createSelector(
  [generationSelector, systemSelector],
  (parameters, system) => {
    const { infillMethod } = parameters;
    const { infillMethods } = system;

    return {
      infillMethod,
      infillMethods,
    };
  },
  defaultSelectorOptions
);

const ParamInfillMethod = () => {
  const dispatch = useAppDispatch();
  const { infillMethod, infillMethods } = useAppSelector(selector);

  const { data: appConfigData } = useGetAppConfigQuery();

  const { t } = useTranslation();

  const handleChange = useCallback(
    (v: string) => {
      dispatch(setInfillMethod(v));
    },
    [dispatch]
  );

  useEffect(() => {
    if (!appConfigData) return;
    if (!appConfigData.patchmatch_enabled) {
      const filteredMethods = infillMethods.filter(
        (method) => method !== 'patchmatch'
      );
      dispatch(setAvailableInfillMethods(filteredMethods));
      dispatch(setInfillMethod(filteredMethods[0]));
    } else {
      dispatch(setInfillMethod('patchmatch'));
    }
  }, [appConfigData, infillMethods, dispatch]);

  return (
    <IAIMantineSelect
      label={t('parameters.infillMethod')}
      value={infillMethod}
      data={infillMethods}
      onChange={handleChange}
    />
  );
};

export default memo(ParamInfillMethod);
