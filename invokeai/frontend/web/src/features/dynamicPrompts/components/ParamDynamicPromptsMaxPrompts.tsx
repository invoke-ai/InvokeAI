import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISlider from 'common/components/IAISlider';
import { maxPromptsChanged, maxPromptsReset } from '../store/slice';
import { createSelector } from '@reduxjs/toolkit';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { useCallback } from 'react';
import { stateSelector } from 'app/store/store';

const selector = createSelector(
  stateSelector,
  (state) => {
    const { maxPrompts, combinatorial } = state.dynamicPrompts;
    const { min, sliderMax, inputMax } =
      state.config.sd.dynamicPrompts.maxPrompts;

    return { maxPrompts, min, sliderMax, inputMax, combinatorial };
  },
  defaultSelectorOptions
);

const ParamDynamicPromptsMaxPrompts = () => {
  const { maxPrompts, min, sliderMax, inputMax, combinatorial } =
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
      isDisabled={!combinatorial}
      min={min}
      max={sliderMax}
      value={maxPrompts}
      onChange={handleChange}
      sliderNumberInputProps={{ max: inputMax }}
      withSliderMarks
      withInput
      inputReadOnly
      withReset
      handleReset={handleReset}
    />
  );
};

export default ParamDynamicPromptsMaxPrompts;
