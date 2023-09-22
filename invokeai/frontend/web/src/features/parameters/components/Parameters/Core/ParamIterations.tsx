import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIInformationalPopover from 'common/components/IAIInformationalPopover';
import IAINumberInput from 'common/components/IAINumberInput';
import IAISlider from 'common/components/IAISlider';
import { setIterations } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createSelector(
  [stateSelector],
  (state) => {
    const { initial, min, sliderMax, inputMax, fineStep, coarseStep } =
      state.config.sd.iterations;
    const { iterations } = state.generation;
    const { shouldUseSliders } = state.ui;

    const step = state.hotkeys.shift ? fineStep : coarseStep;

    return {
      iterations,
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

type Props = {
  asSlider?: boolean;
};

const ParamIterations = ({ asSlider }: Props) => {
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

  return asSlider || shouldUseSliders ? (
    <IAIInformationalPopover details="paramIterations">
      <IAISlider
        label={t('parameters.iterations')}
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
    </IAIInformationalPopover>
  ) : (
    <IAIInformationalPopover
      details="paramIterations"
      buttonLabel="Learn More"
      buttonHref="https://support.invoke.ai/support/solutions/articles/151000159073"
    >
      <IAINumberInput
        label={t('parameters.iterations')}
        step={step}
        min={min}
        max={inputMax}
        onChange={handleChange}
        value={iterations}
        numberInputFieldProps={{ textAlign: 'center' }}
      />
    </IAIInformationalPopover>
  );
};

export default memo(ParamIterations);
