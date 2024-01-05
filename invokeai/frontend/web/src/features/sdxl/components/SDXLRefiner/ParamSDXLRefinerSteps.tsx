import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { setRefinerSteps } from 'features/sdxl/store/sdxlSlice';
import { selectConfigSlice } from 'features/system/store/configSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createMemoizedSelector(selectConfigSlice, (config) => {
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
  const refinerSteps = useAppSelector((s) => s.sdxl.refinerSteps);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const onChange = useCallback(
    (v: number) => {
      dispatch(setRefinerSteps(v));
    },
    [dispatch]
  );

  return (
    <InvControl label={t('sdxl.steps')}>
      <InvSlider
        value={refinerSteps}
        defaultValue={initial}
        min={min}
        max={sliderMax}
        step={step}
        fineStep={fineStep}
        onChange={onChange}
        withNumberInput
        marks={marks}
        numberInputMax={inputMax}
      />
    </InvControl>
  );
};

export default memo(ParamSDXLRefinerSteps);
