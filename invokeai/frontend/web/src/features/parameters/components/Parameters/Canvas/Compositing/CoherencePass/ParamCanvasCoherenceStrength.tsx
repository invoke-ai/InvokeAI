import type { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIInformationalPopover from 'common/components/IAIInformationalPopover/IAIInformationalPopover';
import IAISlider from 'common/components/IAISlider';
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
  const handleReset = useCallback(() => {
    dispatch(setCanvasCoherenceStrength(0.3));
  }, [dispatch]);

  return (
    <IAIInformationalPopover feature="compositingStrength">
      <IAISlider
        label={t('parameters.coherenceStrength')}
        min={0}
        max={1}
        step={0.01}
        sliderNumberInputProps={{ max: 999 }}
        value={canvasCoherenceStrength}
        onChange={handleChange}
        withInput
        withSliderMarks
        withReset
        handleReset={handleReset}
      />
    </IAIInformationalPopover>
  );
};

export default memo(ParamCanvasCoherenceStrength);
