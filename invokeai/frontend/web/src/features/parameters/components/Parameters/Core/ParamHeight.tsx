import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAISlider, { IAIFullSliderProps } from 'common/components/IAISlider';
import { roundToMultiple } from 'common/util/roundDownToMultiple';
import { setHeight, setWidth } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createSelector(
  [stateSelector],
  ({ generation, hotkeys, config }) => {
    const { min, sliderMax, inputMax, fineStep, coarseStep } = config.sd.height;
    const { model, height } = generation;
    const { aspectRatio } = generation;

    const step = hotkeys.shift ? fineStep : coarseStep;

    return {
      model,
      height,
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
  const { model, height, min, sliderMax, inputMax, step, aspectRatio } =
    useAppSelector(selector);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const initial = ['sdxl', 'sdxl-refiner'].includes(model?.base_model as string)
    ? 1024
    : 512;

  const handleChange = useCallback(
    (v: number) => {
      dispatch(setHeight(v));
      if (aspectRatio) {
        const newWidth = roundToMultiple(v * aspectRatio, 8);
        dispatch(setWidth(newWidth));
      }
    },
    [dispatch, aspectRatio]
  );

  const handleReset = useCallback(() => {
    dispatch(setHeight(initial));
    if (aspectRatio) {
      const newWidth = roundToMultiple(initial * aspectRatio, 8);
      dispatch(setWidth(newWidth));
    }
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
