import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvNumberInput } from 'common/components/InvNumberInput/InvNumberInput';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { heightChanged } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createMemoizedSelector(
  [stateSelector],
  ({ generation, config }) => {
    const { min, sliderMax, inputMax, fineStep, coarseStep } = config.sd.height;
    const { model, height } = generation;

    const initial = ['sdxl', 'sdxl-refiner'].includes(
      model?.base_model as string
    )
      ? 1024
      : 512;

    return {
      initial,
      height,
      min,
      max: sliderMax,
      inputMax,
      step: coarseStep,
      fineStep,
    };
  }
);

export const ParamHeight = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const { initial, height, min, max, inputMax, step, fineStep } =
    useAppSelector(selector);

  const onChange = useCallback(
    (v: number) => {
      dispatch(heightChanged(v));
    },
    [dispatch]
  );

  const onReset = useCallback(() => {
    dispatch(heightChanged(initial));
  }, [dispatch, initial]);

  return (
    <InvControl label={t('parameters.height')}>
      <InvSlider
        value={height}
        onChange={onChange}
        onReset={onReset}
        min={min}
        max={max}
        step={step}
        fineStep={fineStep}
        marks={[min, initial, max]}
      />
      <InvNumberInput
        value={height}
        onChange={onChange}
        min={min}
        max={inputMax}
        step={step}
        fineStep={fineStep}
      />
    </InvControl>
  );
});

ParamHeight.displayName = 'ParamHeight';
