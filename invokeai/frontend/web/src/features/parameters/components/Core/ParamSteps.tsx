import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import {
  clampSymmetrySteps,
  setSteps,
} from 'features/parameters/store/generationSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const ParamSteps = () => {
  const steps = useAppSelector((s) => s.generation.steps);
  const initial = useAppSelector((s) => s.config.sd.steps.initial);
  const min = useAppSelector((s) => s.config.sd.steps.min);
  const sliderMax = useAppSelector((s) => s.config.sd.steps.sliderMax);
  const inputMax = useAppSelector((s) => s.config.sd.steps.inputMax);
  const coarseStep = useAppSelector((s) => s.config.sd.steps.coarseStep);
  const fineStep = useAppSelector((s) => s.config.sd.steps.fineStep);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const marks = useMemo(
    () => [min, Math.floor(sliderMax / 2), sliderMax],
    [sliderMax, min]
  );
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
        step={coarseStep}
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
