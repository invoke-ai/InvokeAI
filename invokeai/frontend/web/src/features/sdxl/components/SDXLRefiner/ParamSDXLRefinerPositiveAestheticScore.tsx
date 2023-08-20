import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAISlider from 'common/components/IAISlider';
import { setRefinerPositiveAestheticScore } from 'features/sdxl/store/sdxlSlice';
import { memo, useCallback } from 'react';
import { useIsRefinerAvailable } from 'services/api/hooks/useIsRefinerAvailable';

const selector = createSelector(
  [stateSelector],
  ({ sdxl, hotkeys }) => {
    const { refinerPositiveAestheticScore } = sdxl;
    const { shift } = hotkeys;

    return {
      refinerPositiveAestheticScore,
      shift,
    };
  },
  defaultSelectorOptions
);

const ParamSDXLRefinerPositiveAestheticScore = () => {
  const { refinerPositiveAestheticScore, shift } = useAppSelector(selector);

  const isRefinerAvailable = useIsRefinerAvailable();

  const dispatch = useAppDispatch();

  const handleChange = useCallback(
    (v: number) => dispatch(setRefinerPositiveAestheticScore(v)),
    [dispatch]
  );

  const handleReset = useCallback(
    () => dispatch(setRefinerPositiveAestheticScore(6)),
    [dispatch]
  );

  return (
    <IAISlider
      label="Positive Aesthetic Score"
      step={shift ? 0.1 : 0.5}
      min={1}
      max={10}
      onChange={handleChange}
      handleReset={handleReset}
      value={refinerPositiveAestheticScore}
      sliderNumberInputProps={{ max: 10 }}
      withInput
      withReset
      withSliderMarks
      isInteger={false}
      isDisabled={!isRefinerAvailable}
    />
  );
};

export default memo(ParamSDXLRefinerPositiveAestheticScore);
