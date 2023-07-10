import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAISlider, { IAIFullSliderProps } from 'common/components/IAISlider';
import { roundToMultiple } from 'common/util/roundDownToMultiple';
import { generationSelector } from 'features/parameters/store/generationSelectors';
import { setHeight, setWidth } from 'features/parameters/store/generationSlice';
import { configSelector } from 'features/system/store/configSelectors';
import { hotkeysSelector } from 'features/ui/store/hotkeysSlice';
import { uiSelector } from 'features/ui/store/uiSelectors';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createSelector(
  [generationSelector, hotkeysSelector, configSelector, uiSelector],
  (generation, hotkeys, config, ui) => {
    const { initial, min, sliderMax, inputMax, fineStep, coarseStep } =
      config.sd.width;
    const { width } = generation;
    const { aspectRatio } = ui;

    const step = hotkeys.shift ? fineStep : coarseStep;

    return {
      width,
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

type ParamWidthProps = Omit<IAIFullSliderProps, 'label' | 'value' | 'onChange'>;

const ParamWidth = (props: ParamWidthProps) => {
  const { width, initial, min, sliderMax, inputMax, step, aspectRatio } =
    useAppSelector(selector);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleChange = useCallback(
    (v: number) => {
      dispatch(setWidth(v));
      if (aspectRatio) {
        const newHeight = roundToMultiple(v / aspectRatio, 8);
        dispatch(setHeight(newHeight));
      }
    },
    [dispatch, aspectRatio]
  );

  const handleReset = useCallback(() => {
    dispatch(setWidth(initial));
    if (aspectRatio) {
      const newHeight = roundToMultiple(initial / aspectRatio, 8);
      dispatch(setHeight(newHeight));
    }
  }, [dispatch, initial, aspectRatio]);

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
