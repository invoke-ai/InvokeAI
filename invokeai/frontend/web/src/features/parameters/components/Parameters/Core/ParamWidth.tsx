import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISlider from 'common/components/IAISlider';
import { IAIFullSliderProps } from 'common/components/IAISlider';
import { generationSelector } from 'features/parameters/store/generationSelectors';
import { setWidth } from 'features/parameters/store/generationSlice';
import { configSelector } from 'features/system/store/configSelectors';
import { hotkeysSelector } from 'features/ui/store/hotkeysSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createSelector(
  [generationSelector, hotkeysSelector, configSelector],
  (generation, hotkeys, config) => {
    const { initial, min, sliderMax, inputMax, fineStep, coarseStep } =
      config.sd.width;
    const { width } = generation;

    const step = hotkeys.shift ? fineStep : coarseStep;

    return {
      width,
      initial,
      min,
      sliderMax,
      inputMax,
      step,
    };
  }
);

type ParamWidthProps = Omit<IAIFullSliderProps, 'label' | 'value' | 'onChange'>;

const ParamWidth = (props: ParamWidthProps) => {
  const { width, initial, min, sliderMax, inputMax, step } =
    useAppSelector(selector);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleChange = useCallback(
    (v: number) => {
      dispatch(setWidth(v));
    },
    [dispatch]
  );

  const handleReset = useCallback(() => {
    dispatch(setWidth(initial));
  }, [dispatch, initial]);

  return (
    <IAISlider
      label={t('parameters.width')}
      value={width}
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

export default memo(ParamWidth);
