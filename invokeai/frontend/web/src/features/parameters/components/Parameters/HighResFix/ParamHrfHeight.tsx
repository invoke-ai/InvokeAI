import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAISlider, { IAIFullSliderProps } from 'common/components/IAISlider';
import { roundToMultiple } from 'common/util/roundDownToMultiple';
import {
  setHrfHeight,
  setHrfWidth,
} from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';

const selector = createSelector(
  [stateSelector],
  ({ generation, hotkeys, config }) => {
    const { min, fineStep, coarseStep } = config.sd.height;
    const { model, height, hrfHeight, aspectRatio } = generation;

    const step = hotkeys.shift ? fineStep : coarseStep;

    return {
      model,
      height,
      hrfHeight,
      min,
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

const ParamHrfHeight = (props: ParamHeightProps) => {
  const { model, height, hrfHeight, min, step, aspectRatio } =
    useAppSelector(selector);
  const dispatch = useAppDispatch();

  const initial = ['sdxl', 'sdxl-refiner'].includes(model?.base_model as string)
    ? 1016
    : 504;

  const handleChange = useCallback(
    (v: number) => {
      dispatch(setHrfHeight(v));
      if (aspectRatio) {
        const newWidth = roundToMultiple(v * aspectRatio, 8);
        dispatch(setHrfWidth(newWidth));
      }
    },
    [dispatch, aspectRatio]
  );

  const handleReset = useCallback(() => {
    dispatch(setHrfHeight(initial));
    if (aspectRatio) {
      const newWidth = roundToMultiple(initial * aspectRatio, 8);
      dispatch(setHrfWidth(newWidth));
    }
  }, [dispatch, initial, aspectRatio]);

  return (
    <IAISlider
      label="Initial Height"
      value={hrfHeight}
      min={min}
      step={step}
      max={height}
      onChange={handleChange}
      handleReset={handleReset}
      withInput
      withReset
      withSliderMarks
      sliderNumberInputProps={{ max: height }}
      {...props}
    />
  );
};

export default memo(ParamHrfHeight);
