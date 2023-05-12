import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISlider from 'common/components/IAISlider';
import { setSeamStrength } from 'features/parameters/store/generationSlice';
import { useTranslation } from 'react-i18next';

export default function ParamSeamStrength() {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const seamStrength = useAppSelector(
    (state: RootState) => state.generation.seamStrength
  );

  return (
    <IAISlider
      label={t('parameters.seamStrength')}
      min={0.01}
      max={0.99}
      step={0.01}
      value={seamStrength}
      onChange={(v) => {
        dispatch(setSeamStrength(v));
      }}
      withInput
      withSliderMarks
      withReset
      handleReset={() => {
        dispatch(setSeamStrength(0.7));
      }}
    />
  );
}
