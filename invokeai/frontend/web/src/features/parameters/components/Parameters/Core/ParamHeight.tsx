import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISlider, { IAIFullSliderProps } from 'common/components/IAISlider';
import { generationSelector } from 'features/parameters/store/generationSelectors';
import { setHeight } from 'features/parameters/store/generationSlice';
import { configSelector } from 'features/system/store/configSelectors';
import { hotkeysSelector } from 'features/ui/store/hotkeysSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createSelector(
  [generationSelector, hotkeysSelector, configSelector],
  (generation, hotkeys, config) => {
    const { initial, min, sliderMax, inputMax, fineStep, coarseStep } =
      config.sd.height;
    const { height } = generation;

    const step = hotkeys.shift ? fineStep : coarseStep;

    return {
      height,
      initial,
      min,
      sliderMax,
      inputMax,
      step,
    };
  }
);

type ParamHeightProps = Omit<
  IAIFullSliderProps,
  'label' | 'value' | 'onChange'
>;

const ParamHeight = (props: ParamHeightProps) => {
  const { height, initial, min, sliderMax, inputMax, step } =
    useAppSelector(selector);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleChange = useCallback(
    (v: number) => {
      dispatch(setHeight(v));
    },
    [dispatch]
  );

  const handleReset = useCallback(() => {
    dispatch(setHeight(initial));
  }, [dispatch, initial]);

  return (
    <IAISlider
      label={t('parameters.height')}
      value={height}
      min={min}
      step={step}
      max={sliderMax}
      onChange={handleChange}
      handleReset={handleReset}
      withInput
      withReset
      withSliderMarks
      sliderNumberInputProps={{ max: inputMax }}
      {...props}
    />
  );
};

export default memo(ParamHeight);
