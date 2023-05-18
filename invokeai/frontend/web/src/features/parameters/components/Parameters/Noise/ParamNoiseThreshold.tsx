import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISlider from 'common/components/IAISlider';
import { setThreshold } from 'features/parameters/store/generationSlice';
import { useTranslation } from 'react-i18next';

export default function ParamNoiseThreshold() {
  const dispatch = useAppDispatch();
  const threshold = useAppSelector(
    (state: RootState) => state.generation.threshold
  );
  const { t } = useTranslation();

  return (
    <IAISlider
      label={t('parameters.noiseThreshold')}
      min={0}
      max={20}
      step={0.1}
      onChange={(v) => dispatch(setThreshold(v))}
      handleReset={() => dispatch(setThreshold(0))}
      value={threshold}
      withInput
      withReset
      withSliderMarks
    />
  );
}
