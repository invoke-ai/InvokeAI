import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { setCanvasCoherenceSteps } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamCanvasCoherenceSteps = () => {
  const dispatch = useAppDispatch();
  const canvasCoherenceSteps = useAppSelector(
    (s) => s.generation.canvasCoherenceSteps
  );
  const initial = useAppSelector(
    (s) => s.config.sd.canvasCoherenceSteps.initial
  );
  const sliderMin = useAppSelector(
    (s) => s.config.sd.canvasCoherenceSteps.sliderMin
  );
  const sliderMax = useAppSelector(
    (s) => s.config.sd.canvasCoherenceSteps.sliderMax
  );
  const numberInputMin = useAppSelector(
    (s) => s.config.sd.canvasCoherenceSteps.numberInputMin
  );
  const numberInputMax = useAppSelector(
    (s) => s.config.sd.canvasCoherenceSteps.numberInputMax
  );
  const coarseStep = useAppSelector(
    (s) => s.config.sd.canvasCoherenceSteps.coarseStep
  );
  const fineStep = useAppSelector(
    (s) => s.config.sd.canvasCoherenceSteps.fineStep
  );

  const { t } = useTranslation();

  const handleChange = useCallback(
    (v: number) => {
      dispatch(setCanvasCoherenceSteps(v));
    },
    [dispatch]
  );

  return (
    <InvControl
      label={t('parameters.coherenceSteps')}
      feature="compositingCoherenceSteps"
    >
      <InvSlider
        min={sliderMin}
        max={sliderMax}
        step={coarseStep}
        fineStep={fineStep}
        value={canvasCoherenceSteps}
        defaultValue={initial}
        onChange={handleChange}
        withNumberInput
        numberInputMin={numberInputMin}
        numberInputMax={numberInputMax}
        marks
      />
    </InvControl>
  );
};

export default memo(ParamCanvasCoherenceSteps);
