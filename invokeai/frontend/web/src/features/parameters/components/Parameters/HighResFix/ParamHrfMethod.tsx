import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIMantineSelect from 'common/components/IAIMantineSelect';
import { setHrfMethod } from 'features/parameters/store/generationSlice';
import { ParameterHRFMethod } from 'features/parameters/types/parameterSchemas';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createSelector(
  stateSelector,
  ({ generation }) => {
    const { hrfMethod, hrfEnabled } = generation;
    return { hrfMethod, hrfEnabled };
  },
  defaultSelectorOptions
);

const DATA = ['ESRGAN', 'bilinear'];

// Dropdown selection for the type of high resolution fix method to use.
const ParamHrfMethodSelect = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const { hrfMethod, hrfEnabled } = useAppSelector(selector);

  const handleChange = useCallback(
    (v: ParameterHRFMethod | null) => {
      if (!v) {
        return;
      }
      dispatch(setHrfMethod(v));
    },
    [dispatch]
  );

  return (
    <IAIMantineSelect
      label={t('hrf.upscaleMethod')}
      value={hrfMethod}
      data={DATA}
      onChange={handleChange}
      disabled={!hrfEnabled}
    />
  );
};

export default memo(ParamHrfMethodSelect);
