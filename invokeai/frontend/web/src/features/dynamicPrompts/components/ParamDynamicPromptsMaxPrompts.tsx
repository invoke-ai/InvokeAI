import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAISlider from 'common/components/IAISlider';
import { memo, useCallback } from 'react';
import {
  maxPromptsChanged,
  maxPromptsReset,
} from '../store/dynamicPromptsSlice';

const selector = createSelector(
  stateSelector,
  (state) => {
    const { maxPrompts, combinatorial, isEnabled } = state.dynamicPrompts;
    const { min, sliderMax, inputMax } =
      state.config.sd.dynamicPrompts.maxPrompts;

    return {
      maxPrompts,
      min,
      sliderMax,
      inputMax,
      isDisabled: !isEnabled || !combinatorial,
    };
  },
  defaultSelectorOptions
);

const ParamDynamicPromptsMaxPrompts = () => {
  const { maxPrompts, min, sliderMax, inputMax, isDisabled } =
    useAppSelector(selector);
  const dispatch = useAppDispatch();

  const handleChange = useCallback(
    (v: number) => {
      dispatch(maxPromptsChanged(v));
    },
    [dispatch]
  );

  const handleReset = useCallback(() => {
    dispatch(maxPromptsReset());
  }, [dispatch]);

  return (
    <IAISlider
      label="Max Prompts"
      isDisabled={isDisabled}
      min={min}
      max={sliderMax}
      value={maxPrompts}
      onChange={handleChange}
      sliderNumberInputProps={{ max: inputMax }}
      withSliderMarks
      withInput
      withReset
      handleReset={handleReset}
    />
  );
};

export default memo(ParamDynamicPromptsMaxPrompts);
