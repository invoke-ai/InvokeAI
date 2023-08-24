import type { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISlider from 'common/components/IAISlider';
import { setCanvasCoherenceSteps } from 'features/parameters/store/generationSlice';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const ParamCanvasCoherenceSteps = () => {
  const dispatch = useAppDispatch();
  const canvasCoherenceSteps = useAppSelector(
    (state: RootState) => state.generation.canvasCoherenceSteps
  );
  const { t } = useTranslation();

  return (
    <IAISlider
      label={t('parameters.coherenceSteps')}
      min={1}
      max={100}
      step={1}
      sliderNumberInputProps={{ max: 999 }}
      value={canvasCoherenceSteps}
      onChange={(v) => {
        dispatch(setCanvasCoherenceSteps(v));
      }}
      withInput
      withSliderMarks
      withReset
      handleReset={() => {
        dispatch(setCanvasCoherenceSteps(20));
      }}
    />
  );
};

export default memo(ParamCanvasCoherenceSteps);
