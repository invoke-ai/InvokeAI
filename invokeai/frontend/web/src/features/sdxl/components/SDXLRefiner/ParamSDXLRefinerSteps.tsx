import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { setRefinerSteps } from 'features/sdxl/store/sdxlSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createMemoizedSelector([stateSelector], ({ config }) => {
  const { initial, min, sliderMax, inputMax, fineStep, coarseStep } =
    config.sd.steps;

  return {
    marks: [min, Math.floor(sliderMax / 2), sliderMax],
    initial,
    min,
    sliderMax,
    inputMax,
    step: coarseStep,
    fineStep,
  };
});
const ParamSDXLRefinerSteps = () => {
  const { initial, min, sliderMax, inputMax, step, fineStep, marks } =
    useAppSelector(selector);
  const refinerSteps = useAppSelector((state) => state.sdxl.refinerSteps);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const onChange = useCallback(
    (v: number) => {
      dispatch(setRefinerSteps(v));
    },
    [dispatch]
  );

  const onReset = useCallback(() => {
    dispatch(setRefinerSteps(initial));
  }, [dispatch, initial]);

  return (
    <InvControl label={t('sdxl.steps')}>
      <InvSlider
        value={refinerSteps}
        min={min}
        max={sliderMax}
        step={step}
        fineStep={fineStep}
        onChange={onChange}
        onReset={onReset}
        withNumberInput
        marks={marks}
        numberInputMax={inputMax}
      />
    </InvControl>
  );
};

export default memo(ParamSDXLRefinerSteps);
