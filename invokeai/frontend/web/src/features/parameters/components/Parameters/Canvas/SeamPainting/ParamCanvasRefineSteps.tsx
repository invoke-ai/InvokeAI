import type { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISlider from 'common/components/IAISlider';
import { setCanvasRefineSteps } from 'features/parameters/store/generationSlice';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const ParamCanvasRefineSteps = () => {
  const dispatch = useAppDispatch();
  const canvasRefineSteps = useAppSelector(
    (state: RootState) => state.generation.canvasRefineSteps
  );
  const { t } = useTranslation();

  return (
    <IAISlider
      label={t('parameters.refineSteps')}
      min={1}
      max={100}
      step={1}
      sliderNumberInputProps={{ max: 999 }}
      value={canvasRefineSteps}
      onChange={(v) => {
        dispatch(setCanvasRefineSteps(v));
      }}
      withInput
      withSliderMarks
      withReset
      handleReset={() => {
        dispatch(setCanvasRefineSteps(20));
      }}
    />
  );
};

export default memo(ParamCanvasRefineSteps);
