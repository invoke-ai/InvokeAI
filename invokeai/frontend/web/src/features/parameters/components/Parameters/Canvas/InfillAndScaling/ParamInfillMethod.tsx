import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAISelect from 'common/components/IAISelect';
import { generationSelector } from 'features/parameters/store/generationSelectors';
import { setInfillMethod } from 'features/parameters/store/generationSlice';
import { systemSelector } from 'features/system/store/systemSelectors';

import { ChangeEvent, memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

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

  const { t } = useTranslation();

  const handleChange = useCallback(
    (e: ChangeEvent<HTMLSelectElement>) => {
      dispatch(setInfillMethod(e.target.value));
    },
    [dispatch]
  );

  return (
    <IAISelect
      label={t('parameters.infillMethod')}
      value={infillMethod}
      validValues={infillMethods}
      onChange={handleChange}
    />
  );
};

export default memo(ParamInfillMethod);
