import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISlider from 'common/components/IAISlider';
import { setHorizontalSymmetrySteps } from 'features/parameters/store/generationSlice';
import { useTranslation } from 'react-i18next';

export default function ParamSymmetryHorizontal() {
  const horizontalSymmetrySteps = useAppSelector(
    (state: RootState) => state.generation.horizontalSymmetrySteps
  );

  const steps = useAppSelector((state: RootState) => state.generation.steps);

  const dispatch = useAppDispatch();

  const { t } = useTranslation();

  return (
    <IAISlider
      label={t('parameters.hSymmetryStep')}
      value={horizontalSymmetrySteps}
      onChange={(v) => dispatch(setHorizontalSymmetrySteps(v))}
      min={0}
      max={steps}
      step={1}
      withInput
      withSliderMarks
      withReset
      handleReset={() => dispatch(setHorizontalSymmetrySteps(0))}
    />
  );
}
