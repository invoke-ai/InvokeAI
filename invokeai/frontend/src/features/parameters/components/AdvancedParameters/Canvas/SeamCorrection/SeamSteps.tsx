import type { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAISlider from 'common/components/IAISlider';
import { setSeamSteps } from 'features/parameters/store/generationSlice';
import { useTranslation } from 'react-i18next';

export default function SeamSteps() {
  const { t } = useTranslation();
  const seamSteps = useAppSelector(
    (state: RootState) => state.generation.seamSteps
  );
  const dispatch = useAppDispatch();

  return (
    <IAISlider
      sliderMarkRightOffset={-4}
      label={t('parameters.seamSteps')}
      min={1}
      max={100}
      sliderNumberInputProps={{ max: 999 }}
      value={seamSteps}
      onChange={(v) => {
        dispatch(setSeamSteps(v));
      }}
      withInput
      withSliderMarks
      withReset
      handleReset={() => {
        dispatch(setSeamSteps(30));
      }}
    />
  );
}
