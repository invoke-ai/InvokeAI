import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAISlider, { IAIFullSliderProps } from 'common/components/IAISlider';
import { generationSelector } from 'features/parameters/store/generationSelectors';
import { setHeight, setWidth } from 'features/parameters/store/generationSlice';
import { configSelector } from 'features/system/store/configSelectors';
import { hotkeysSelector } from 'features/ui/store/hotkeysSlice';
import { uiSelector } from 'features/ui/store/uiSelectors';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { roundToEight } from './ParamAspectRatio';

const selector = createSelector(
  [generationSelector, hotkeysSelector, configSelector, uiSelector],
  (generation, hotkeys, config, ui) => {
    const { initial, min, sliderMax, inputMax, fineStep, coarseStep } =
      config.sd.height;
    const { height } = generation;
    const { aspectRatio } = ui;

    const step = hotkeys.shift ? fineStep : coarseStep;

    return {
      height,
      initial,
      min,
      sliderMax,
      inputMax,
      step,
      aspectRatio,
    };
  },
  defaultSelectorOptions
);

type ParamHeightProps = Omit<
  IAIFullSliderProps,
  'label' | 'value' | 'onChange'
>;

const ParamHeight = (props: ParamHeightProps) => {
  const { height, initial, min, sliderMax, inputMax, step, aspectRatio } =
    useAppSelector(selector);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleChange = useCallback(
    (v: number) => {
      dispatch(setHeight(v));
      if (aspectRatio) dispatch(setWidth(roundToEight(height * aspectRatio)));
    },
    [dispatch, height, aspectRatio]
  );

  const handleReset = useCallback(() => {
    dispatch(setHeight(initial));
    if (aspectRatio) dispatch(setWidth(roundToEight(initial * aspectRatio)));
  }, [dispatch, initial, aspectRatio]);

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
