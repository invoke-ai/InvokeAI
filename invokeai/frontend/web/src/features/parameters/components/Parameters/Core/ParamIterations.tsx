import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAINumberInput from 'common/components/IAINumberInput';
import IAISlider from 'common/components/IAISlider';
import { generationSelector } from 'features/parameters/store/generationSelectors';
import { setIterations } from 'features/parameters/store/generationSlice';
import { configSelector } from 'features/system/store/configSelectors';
import { hotkeysSelector } from 'features/ui/store/hotkeysSlice';
import { uiSelector } from 'features/ui/store/uiSelectors';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createSelector(
  [generationSelector, configSelector, uiSelector, hotkeysSelector],
  (generation, config, ui, hotkeys) => {
    const { initial, min, sliderMax, inputMax, fineStep, coarseStep } =
      config.sd.iterations;
    const { iterations } = generation;
    const { shouldUseSliders } = ui;

    const step = hotkeys.shift ? fineStep : coarseStep;

    return {
      iterations,
      initial,
      min,
      sliderMax,
      inputMax,
      step,
      shouldUseSliders,
    };
  }
);

const ParamIterations = () => {
  const {
    iterations,
    initial,
    min,
    sliderMax,
    inputMax,
    step,
    shouldUseSliders,
  } = useAppSelector(selector);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleChange = useCallback(
    (v: number) => {
      dispatch(setIterations(v));
    },
    [dispatch]
  );

  const handleReset = useCallback(() => {
    dispatch(setIterations(initial));
  }, [dispatch, initial]);

  return shouldUseSliders ? (
    <IAISlider
      label={t('parameters.images')}
      step={step}
      min={min}
      max={sliderMax}
      onChange={handleChange}
      handleReset={handleReset}
      value={iterations}
      withInput
      withReset
      withSliderMarks
      sliderNumberInputProps={{ max: inputMax }}
    />
  ) : (
    <IAINumberInput
      label={t('parameters.images')}
      step={step}
      min={min}
      max={inputMax}
      onChange={handleChange}
      value={iterations}
      numberInputFieldProps={{ textAlign: 'center' }}
    />
  );
};

export default memo(ParamIterations);
