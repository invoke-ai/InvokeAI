import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISlider from 'common/components/IAISlider';
import { setVerticalSymmetrySteps } from 'features/parameters/store/generationSlice';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export default function ParamSymmetryVertical() {
  const verticalSymmetrySteps = useAppSelector(
    (state: RootState) => state.generation.verticalSymmetrySteps
  );

  const steps = useAppSelector((state: RootState) => state.generation.steps);

  const dispatch = useAppDispatch();

  const { t } = useTranslation();

  const handleChange = useCallback(
    (v: number) => {
      dispatch(setVerticalSymmetrySteps(v));
    },
    [dispatch]
  );
  const handleReset = useCallback(() => {
    dispatch(setVerticalSymmetrySteps(0));
  }, [dispatch]);

  return (
    <IAISlider
      label={t('parameters.vSymmetryStep')}
      value={verticalSymmetrySteps}
      onChange={handleChange}
      min={0}
      max={steps}
      step={1}
      withInput
      withSliderMarks
      withReset
      handleReset={handleReset}
    />
  );
}
