import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIMantineSelect from 'common/components/IAIMantineSelect';
import { setHrfMethod } from 'features/parameters/store/generationSlice';
import { HrfMethodParam } from 'features/parameters/types/parameterSchemas';
import { memo, useCallback } from 'react';

const selector = createSelector(
  stateSelector,
  ({ generation }) => {
    const { hrfMethod } = generation;
    return { hrfMethod };
  },
  defaultSelectorOptions
);

const DATA = ['ERSGAN', 'bilinear'];

// Dropdown selection for the type of high resolution fix method to use.
const ParamHrfMethodSelect = () => {
  const dispatch = useAppDispatch();
  const { hrfMethod } = useAppSelector(selector);

  const handleChange = useCallback(
    (v: HrfMethodParam | null) => {
      if (!v) {
        return;
      }
      dispatch(setHrfMethod(v));
    },
    [dispatch]
  );

  return (
    <IAIMantineSelect
      label="Method"
      value={hrfMethod}
      data={DATA}
      onChange={handleChange}
    />
  );
};

export default memo(ParamHrfMethodSelect);
