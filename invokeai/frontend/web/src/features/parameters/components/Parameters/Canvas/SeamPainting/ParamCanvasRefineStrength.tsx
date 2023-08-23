import type { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISlider from 'common/components/IAISlider';
import { setCanvasRefineStrength } from 'features/parameters/store/generationSlice';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const ParamCanvasRefineStrength = () => {
  const dispatch = useAppDispatch();
  const canvasRefineStrength = useAppSelector(
    (state: RootState) => state.generation.canvasRefineStrength
  );
  const { t } = useTranslation();

  return (
    <IAISlider
      label={t('parameters.refineStrength')}
      min={0}
      max={1}
      step={0.01}
      sliderNumberInputProps={{ max: 999 }}
      value={canvasRefineStrength}
      onChange={(v) => {
        dispatch(setCanvasRefineStrength(v));
      }}
      withInput
      withSliderMarks
      withReset
      handleReset={() => {
        dispatch(setCanvasRefineStrength(0.3));
      }}
    />
  );
};

export default memo(ParamCanvasRefineStrength);
