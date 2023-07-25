import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAISlider from 'common/components/IAISlider';
import { setRefinerStart } from 'features/sdxl/store/sdxlSlice';
import { memo, useCallback } from 'react';

const selector = createSelector(
  [stateSelector],
  ({ sdxl, hotkeys }) => {
    const { refinerStart } = sdxl;
    const { shift } = hotkeys;

    return {
      refinerStart,
      shift,
    };
  },
  defaultSelectorOptions
);

const ParamSDXLRefinerStart = () => {
  const { refinerStart, shift } = useAppSelector(selector);
  const dispatch = useAppDispatch();

  const handleChange = useCallback(
    (v: number) => dispatch(setRefinerStart(v)),
    [dispatch]
  );

  const handleReset = useCallback(
    () => dispatch(setRefinerStart(0.7)),
    [dispatch]
  );

  return (
    <IAISlider
      label="Refiner Start"
      step={shift ? 0.1 : 0.01}
      min={0.01}
      max={1}
      onChange={handleChange}
      handleReset={handleReset}
      value={refinerStart}
      sliderNumberInputProps={{ max: 1 }}
      withInput
      withReset
      withSliderMarks
      isInteger={false}
    />
  );
};

export default memo(ParamSDXLRefinerStart);
