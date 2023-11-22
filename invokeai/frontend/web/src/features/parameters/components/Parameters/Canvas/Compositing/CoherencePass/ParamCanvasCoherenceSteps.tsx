import type { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIInformationalPopover from 'common/components/IAIInformationalPopover/IAIInformationalPopover';
import IAISlider from 'common/components/IAISlider';
import { setCanvasCoherenceSteps } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamCanvasCoherenceSteps = () => {
  const dispatch = useAppDispatch();
  const canvasCoherenceSteps = useAppSelector(
    (state: RootState) => state.generation.canvasCoherenceSteps
  );
  const { t } = useTranslation();

  const handleChange = useCallback(
    (v: number) => {
      dispatch(setCanvasCoherenceSteps(v));
    },
    [dispatch]
  );

  const handleReset = useCallback(() => {
    dispatch(setCanvasCoherenceSteps(20));
  }, [dispatch]);

  return (
    <IAIInformationalPopover feature="compositingCoherenceSteps">
      <IAISlider
        label={t('parameters.coherenceSteps')}
        min={1}
        max={100}
        step={1}
        sliderNumberInputProps={{ max: 999 }}
        value={canvasCoherenceSteps}
        onChange={handleChange}
        withInput
        withSliderMarks
        withReset
        handleReset={handleReset}
      />
    </IAIInformationalPopover>
  );
};

export default memo(ParamCanvasCoherenceSteps);
