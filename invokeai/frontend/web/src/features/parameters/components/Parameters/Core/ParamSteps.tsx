import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAINumberInput from 'common/components/IAINumberInput';

import IAISlider from 'common/components/IAISlider';
import {
  clampSymmetrySteps,
  setSteps,
} from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createSelector(
  [stateSelector],
  ({ generation, config, ui, hotkeys }) => {
    const { initial, min, sliderMax, inputMax, fineStep, coarseStep } =
      config.sd.steps;
    const { steps } = generation;
    const { shouldUseSliders } = ui;

    const step = hotkeys.shift ? fineStep : coarseStep;

    return {
      steps,
      initial,
      min,
      sliderMax,
      inputMax,
      step,
      shouldUseSliders,
    };
  },
  defaultSelectorOptions
);

const ParamSteps = () => {
  const { steps, initial, min, sliderMax, inputMax, step, shouldUseSliders } =
    useAppSelector(selector);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleChange = useCallback(
    (v: number) => {
      dispatch(setSteps(v));
    },
    [dispatch]
  );
  const handleReset = useCallback(() => {
    dispatch(setSteps(initial));
  }, [dispatch, initial]);

  const handleBlur = useCallback(() => {
    dispatch(clampSymmetrySteps());
  }, [dispatch]);

  return shouldUseSliders ? (
    <IAISlider
      label={t('parameters.steps')}
      min={min}
      max={sliderMax}
      step={step}
      onChange={handleChange}
      handleReset={handleReset}
      value={steps}
      withInput
      withReset
      withSliderMarks
      sliderNumberInputProps={{ max: inputMax }}
    />
  ) : (
    <IAINumberInput
      label={t('parameters.steps')}
      min={min}
      max={inputMax}
      step={step}
      onChange={handleChange}
      value={steps}
      numberInputFieldProps={{ textAlign: 'center' }}
      onBlur={handleBlur}
    />
  );
};

export default memo(ParamSteps);
