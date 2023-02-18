import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAISlider from 'common/components/IAISlider';
import { setSeamStrength } from 'features/parameters/store/generationSlice';
import { useTranslation } from 'react-i18next';

export default function SeamStrength() {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const seamStrength = useAppSelector(
    (state: RootState) => state.generation.seamStrength
  );

  return (
    <IAISlider
      sliderMarkRightOffset={-7}
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
