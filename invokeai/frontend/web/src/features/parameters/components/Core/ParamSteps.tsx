import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import {
  clampSymmetrySteps,
  selectGenerationSlice,
  setSteps,
} from 'features/parameters/store/generationSlice';
import { selectConfigSlice } from 'features/system/store/configSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createMemoizedSelector(
  selectGenerationSlice,
  selectConfigSlice,
  (generation, config) => {
    const { initial, min, sliderMax, inputMax, fineStep, coarseStep } =
      config.sd.steps;
    const { steps } = generation;

    return {
      marks: [min, Math.floor(sliderMax / 2), sliderMax],
      steps,
      initial,
      min,
      sliderMax,
      inputMax,
      step: coarseStep,
      fineStep,
    };
  }
);

const ParamSteps = () => {
  const { steps, initial, min, sliderMax, inputMax, step, fineStep, marks } =
    useAppSelector(selector);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const onChange = useCallback(
    (v: number) => {
      dispatch(setSteps(v));
    },
    [dispatch]
  );

  const onBlur = useCallback(() => {
    dispatch(clampSymmetrySteps());
  }, [dispatch]);

  return (
    <InvControl label={t('parameters.steps')} feature="paramSteps">
      <InvSlider
        value={steps}
        defaultValue={initial}
        min={min}
        max={sliderMax}
        step={step}
        fineStep={fineStep}
        onChange={onChange}
        onBlur={onBlur}
        withNumberInput
        marks={marks}
        numberInputMax={inputMax}
      />
    </InvControl>
  );
};

export default memo(ParamSteps);
