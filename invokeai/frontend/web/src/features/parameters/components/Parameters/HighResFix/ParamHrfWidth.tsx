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

function findPrevMultipleOfEight(n: number): number {
  return Math.floor((n - 1) / 8) * 8;
}

const selector = createSelector(
  [stateSelector],
  ({ generation, hotkeys, config }) => {
    const { min, fineStep, coarseStep } = config.sd.width;
    const { model, width, hrfWidth, aspectRatio, hrfEnabled } = generation;

    const step = hotkeys.shift ? fineStep : coarseStep;

    return {
      model,
      width,
      hrfWidth,
      min,
      step,
      aspectRatio,
      hrfEnabled,
    };
  },
  defaultSelectorOptions
);

type ParamWidthProps = Omit<IAIFullSliderProps, 'label' | 'value' | 'onChange'>;

const ParamHrfWidth = (props: ParamWidthProps) => {
  const { width, hrfWidth, min, step, aspectRatio, hrfEnabled } =
    useAppSelector(selector);
  const dispatch = useAppDispatch();
  const maxHrfWidth = Math.max(findPrevMultipleOfEight(width), min);

  const handleChange = useCallback(
    (v: number) => {
      dispatch(setHrfWidth(v));
      if (aspectRatio) {
        const newHeight = roundToMultiple(v / aspectRatio, 8);
        dispatch(setHrfHeight(newHeight));
      }
    },
    [dispatch, aspectRatio]
  );

  const handleReset = useCallback(() => {
    dispatch(setHrfWidth(maxHrfWidth));
    if (aspectRatio) {
      const newHeight = roundToMultiple(maxHrfWidth / aspectRatio, 8);
      dispatch(setHrfHeight(newHeight));
    }
  }, [dispatch, maxHrfWidth, aspectRatio]);

  return (
    <IAISlider
      label="Initial Width"
      value={hrfWidth}
      min={min}
      step={step}
      max={maxHrfWidth}
      onChange={handleChange}
      handleReset={handleReset}
      withInput
      withReset
      withSliderMarks
      sliderNumberInputProps={{ max: maxHrfWidth }}
      isDisabled={!hrfEnabled}
      {...props}
    />
  );
};

export default memo(ParamHrfWidth);
