import type { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { setCanvasCoherenceStrength } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamCanvasCoherenceStrength = () => {
  const dispatch = useAppDispatch();
  const canvasCoherenceStrength = useAppSelector(
    (state: RootState) => state.generation.canvasCoherenceStrength
  );
  const { t } = useTranslation();

  const handleChange = useCallback(
    (v: number) => {
      dispatch(setCanvasCoherenceStrength(v));
    },
    [dispatch]
  );

  return (
    <InvControl
      label={t('parameters.coherenceStrength')}
      feature="compositingStrength"
    >
      <InvSlider
        min={0}
        max={1}
        step={0.01}
        value={canvasCoherenceStrength}
        defaultValue={0.75}
        onChange={handleChange}
        withNumberInput
        numberInputMax={999}
        marks
      />
    </InvControl>
  );
};

export default memo(ParamCanvasCoherenceStrength);
