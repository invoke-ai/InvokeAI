import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAISlider from 'common/components/IAISlider';
import { setRefinerStart } from 'features/sdxl/store/sdxlSlice';
import { memo, useCallback } from 'react';
import { useIsRefinerAvailable } from 'services/api/hooks/useIsRefinerAvailable';

const selector = createSelector(
  [stateSelector],
  ({ sdxl }) => {
    const { refinerStart } = sdxl;
    return {
      refinerStart,
    };
  },
  defaultSelectorOptions
);

const ParamSDXLRefinerStart = () => {
  const { refinerStart } = useAppSelector(selector);
  const dispatch = useAppDispatch();
  const isRefinerAvailable = useIsRefinerAvailable();
  const handleChange = useCallback(
    (v: number) => dispatch(setRefinerStart(v)),
    [dispatch]
  );

  const handleReset = useCallback(
    () => dispatch(setRefinerStart(0.8)),
    [dispatch]
  );

  return (
    <IAISlider
      label="Refiner Start"
      step={0.01}
      min={0}
      max={1}
      onChange={handleChange}
      handleReset={handleReset}
      value={refinerStart}
      withInput
      withReset
      withSliderMarks
      isInteger={false}
      isDisabled={!isRefinerAvailable}
    />
  );
};

export default memo(ParamSDXLRefinerStart);
