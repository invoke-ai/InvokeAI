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
        min={1}
        max={100}
        step={1}
        value={canvasCoherenceSteps}
        defaultValue={20}
        onChange={handleChange}
        withNumberInput
        numberInputMax={999}
        marks
      />
    </InvControl>
  );
};

export default memo(ParamCanvasCoherenceSteps);
