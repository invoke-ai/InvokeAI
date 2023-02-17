import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAISlider from 'common/components/IAISlider';
import { setThreshold } from 'features/parameters/store/generationSlice';
import { useTranslation } from 'react-i18next';

export default function Threshold() {
  const dispatch = useAppDispatch();
  const threshold = useAppSelector(
    (state: RootState) => state.generation.threshold
  );
  const { t } = useTranslation();

  return (
    <IAISlider
      label={t('parameters:noiseThreshold')}
      min={0}
      max={1}
      step={0.005}
      onChange={(v) => dispatch(setThreshold(v))}
      handleReset={() => dispatch(setThreshold(0))}
      value={threshold}
      withInput
      withReset
      withSliderMarks
      inputWidth="6rem"
    />
  );
}
