import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvNumberInput } from 'common/components/InvNumberInput/InvNumberInput';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { widthChanged } from 'features/parameters/store/generationSlice';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createMemoizedSelector(
  [stateSelector],
  ({ generation, config }) => {
    const { min, sliderMax, inputMax, fineStep, coarseStep } = config.sd.width;
    const { model, width } = generation;

    const initial = ['sdxl', 'sdxl-refiner'].includes(
      model?.base_model as string
    )
      ? 1024
      : 512;

    return {
      initial,
      width,
      min,
      max: sliderMax,
      step: coarseStep,
      inputMax,
      fineStep,
    };
  }
);
export const ParamWidth = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const { initial, width, min, max, inputMax, step, fineStep } =
    useAppSelector(selector);

  const onChange = useCallback(
    (v: number) => {
      dispatch(widthChanged(v));
    },
    [dispatch]
  );

  const onReset = useCallback(() => {
    dispatch(widthChanged(initial));
  }, [dispatch, initial]);

  return (
    <InvControl label={t('parameters.width')}>
      <InvSlider
        value={width}
        onChange={onChange}
        onReset={onReset}
        min={min}
        max={max}
        step={step}
        fineStep={fineStep}
        marks={[min, initial, max]}
      />
      <InvNumberInput
        value={width}
        onChange={onChange}
        min={min}
        max={inputMax}
        step={step}
        fineStep={fineStep}
      />
    </InvControl>
  );
};
