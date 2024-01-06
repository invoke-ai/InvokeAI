import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { setRefinerSteps } from 'features/sdxl/store/sdxlSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const ParamSDXLRefinerSteps = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const refinerSteps = useAppSelector((s) => s.sdxl.refinerSteps);
  const initial = useAppSelector((s) => s.config.sd.steps.initial);
  const min = useAppSelector((s) => s.config.sd.steps.min);
  const sliderMax = useAppSelector((s) => s.config.sd.steps.sliderMax);
  const inputMax = useAppSelector((s) => s.config.sd.steps.inputMax);
  const coarseStep = useAppSelector((s) => s.config.sd.steps.coarseStep);
  const fineStep = useAppSelector((s) => s.config.sd.steps.fineStep);

  const marks = useMemo(
    () => [min, Math.floor(sliderMax / 2), sliderMax],
    [sliderMax, min]
  );

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
        step={coarseStep}
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
