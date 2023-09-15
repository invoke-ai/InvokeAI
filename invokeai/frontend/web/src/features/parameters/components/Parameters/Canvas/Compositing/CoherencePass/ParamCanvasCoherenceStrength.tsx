import type { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISlider from 'common/components/IAISlider';
import { CompositingStrengthPopover } from 'features/informationalPopovers/components/compositingStrength';
import { setCanvasCoherenceStrength } from 'features/parameters/store/generationSlice';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const ParamCanvasCoherenceStrength = () => {
  const dispatch = useAppDispatch();
  const canvasCoherenceStrength = useAppSelector(
    (state: RootState) => state.generation.canvasCoherenceStrength
  );
  const { t } = useTranslation();

  return (
    <CompositingStrengthPopover>
      <IAISlider
        label={t('parameters.coherenceStrength')}
        min={0}
        max={1}
        step={0.01}
        sliderNumberInputProps={{ max: 999 }}
        value={canvasCoherenceStrength}
        onChange={(v) => {
          dispatch(setCanvasCoherenceStrength(v));
        }}
        withInput
        withSliderMarks
        withReset
        handleReset={() => {
          dispatch(setCanvasCoherenceStrength(0.3));
        }}
      />
    </CompositingStrengthPopover>
  );
};

export default memo(ParamCanvasCoherenceStrength);
