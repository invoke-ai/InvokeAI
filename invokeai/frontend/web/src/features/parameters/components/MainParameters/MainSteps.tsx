import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAINumberInput from 'common/components/IAINumberInput';

import IAISlider from 'common/components/IAISlider';
import {
  clampSymmetrySteps,
  setSteps,
} from 'features/parameters/store/generationSlice';
import { useTranslation } from 'react-i18next';

export default function MainSteps() {
  const dispatch = useAppDispatch();
  const steps = useAppSelector((state: RootState) => state.generation.steps);
  const shouldUseSliders = useAppSelector(
    (state: RootState) => state.ui.shouldUseSliders
  );
  const { t } = useTranslation();

  const handleChangeSteps = (v: number) => {
    dispatch(setSteps(v));
  };

  const handleBlur = () => {
    dispatch(clampSymmetrySteps());
  };

  return shouldUseSliders ? (
    <IAISlider
      label={t('parameters.steps')}
      min={1}
      step={1}
      onChange={handleChangeSteps}
      handleReset={() => dispatch(setSteps(20))}
      value={steps}
      withInput
      withReset
      withSliderMarks
      sliderNumberInputProps={{ max: 9999 }}
    />
  ) : (
    <IAINumberInput
      label={t('parameters.steps')}
      min={1}
      max={9999}
      step={1}
      onChange={handleChangeSteps}
      value={steps}
      numberInputFieldProps={{ textAlign: 'center' }}
      onBlur={handleBlur}
    />
  );
}
