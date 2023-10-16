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
import { memo, useCallback, useEffect } from 'react';

function findPrevMultipleOfEight(n: number): number {
  return Math.floor((n - 1) / 8) * 8;
}

const selector = createSelector(
  [stateSelector],
  ({ generation, hotkeys, config }) => {
    const { min, fineStep, coarseStep } = config.sd.height;
    const { model, height, hrfHeight, aspectRatio, hrfManualResEnabled } =
      generation;

    const step = hotkeys.shift ? fineStep : coarseStep;

    return {
      model,
      height,
      hrfHeight,
      min,
      step,
      aspectRatio,
      hrfManualResEnabled,
    };
  },
  defaultSelectorOptions
);

type ParamHeightProps = Omit<
  IAIFullSliderProps,
  'label' | 'value' | 'onChange'
>;

const ParamHrfHeight = (props: ParamHeightProps) => {
  const { height, hrfHeight, min, step, aspectRatio, hrfManualResEnabled } =
    useAppSelector(selector);
  const dispatch = useAppDispatch();
  const maxHrfHeight = Math.max(findPrevMultipleOfEight(height), min);

  // Makes sure the slider never goes above its max.
  useEffect(() => {
    if (hrfHeight > maxHrfHeight) {
      dispatch(setHrfHeight(maxHrfHeight));
    }
  }, [dispatch, hrfHeight, maxHrfHeight]);

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

  return (
    <IAISlider
      label="Initial Height"
      value={hrfHeight}
      min={min}
      step={step}
      max={maxHrfHeight}
      onChange={handleChange}
      withInput
      withSliderMarks
      sliderNumberInputProps={{ max: maxHrfHeight }}
      isDisabled={!hrfManualResEnabled}
      {...props}
    />
  );
};

export default memo(ParamHrfHeight);
