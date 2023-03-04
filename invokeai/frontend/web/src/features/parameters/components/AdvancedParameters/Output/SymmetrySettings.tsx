import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAISlider from 'common/components/IAISlider';
import {
  setHorizontalSymmetryTimePercentage,
  setVerticalSymmetryTimePercentage,
} from 'features/parameters/store/generationSlice';
import { useTranslation } from 'react-i18next';

export default function SymmetrySettings() {
  const horizontalSymmetryTimePercentage = useAppSelector(
    (state: RootState) => state.generation.horizontalSymmetryTimePercentage
  );

  const verticalSymmetryTimePercentage = useAppSelector(
    (state: RootState) => state.generation.verticalSymmetryTimePercentage
  );

  const steps = useAppSelector((state: RootState) => state.generation.steps);

  const dispatch = useAppDispatch();

  const { t } = useTranslation();

  return (
    <>
      <IAISlider
        label={t('parameters.hSymmetryStep')}
        value={horizontalSymmetryTimePercentage}
        onChange={(v) => dispatch(setHorizontalSymmetryTimePercentage(v))}
        min={0}
        max={steps}
        step={1}
        withInput
        withSliderMarks
        withReset
        handleReset={() => dispatch(setHorizontalSymmetryTimePercentage(0))}
        sliderMarkRightOffset={-6}
      ></IAISlider>
      <IAISlider
        label={t('parameters.vSymmetryStep')}
        value={verticalSymmetryTimePercentage}
        onChange={(v) => dispatch(setVerticalSymmetryTimePercentage(v))}
        min={0}
        max={steps}
        step={1}
        withInput
        withSliderMarks
        withReset
        handleReset={() => dispatch(setVerticalSymmetryTimePercentage(0))}
        sliderMarkRightOffset={-6}
      ></IAISlider>
    </>
  );
}
