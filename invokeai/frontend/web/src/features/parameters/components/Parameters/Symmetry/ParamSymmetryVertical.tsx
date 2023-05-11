import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISlider from 'common/components/IAISlider';
import { setVerticalSymmetrySteps } from 'features/parameters/store/generationSlice';
import { useTranslation } from 'react-i18next';

export default function ParamSymmetryVertical() {
  const verticalSymmetrySteps = useAppSelector(
    (state: RootState) => state.generation.verticalSymmetrySteps
  );

  const steps = useAppSelector((state: RootState) => state.generation.steps);

  const dispatch = useAppDispatch();

  const { t } = useTranslation();

  return (
    <IAISlider
      label={t('parameters.vSymmetryStep')}
      value={verticalSymmetrySteps}
      onChange={(v) => dispatch(setVerticalSymmetrySteps(v))}
      min={0}
      max={steps}
      step={1}
      withInput
      withSliderMarks
      withReset
      handleReset={() => dispatch(setVerticalSymmetrySteps(0))}
    />
  );
}
