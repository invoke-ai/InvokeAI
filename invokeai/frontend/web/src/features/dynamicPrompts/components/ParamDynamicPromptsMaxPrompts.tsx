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
import { useTranslation } from 'react-i18next';
import IAIInformationalPopover from 'common/components/IAIInformationalPopover/IAIInformationalPopover';

const selector = createSelector(
  stateSelector,
  (state) => {
    const { maxPrompts, combinatorial } = state.dynamicPrompts;
    const { min, sliderMax, inputMax } =
      state.config.sd.dynamicPrompts.maxPrompts;

    return {
      maxPrompts,
      min,
      sliderMax,
      inputMax,
      isDisabled: !combinatorial,
    };
  },
  defaultSelectorOptions
);

const ParamDynamicPromptsMaxPrompts = () => {
  const { maxPrompts, min, sliderMax, inputMax, isDisabled } =
    useAppSelector(selector);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

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
    <IAIInformationalPopover feature="dynamicPromptsMaxPrompts">
      <IAISlider
        label={t('dynamicPrompts.maxPrompts')}
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
    </IAIInformationalPopover>
  );
};

export default memo(ParamDynamicPromptsMaxPrompts);
