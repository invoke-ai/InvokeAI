import type { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISlider from 'common/components/IAISlider';
import { setCodeformerFidelity } from 'features/parameters/store/postprocessingSlice';
import { useTranslation } from 'react-i18next';

export default function CodeformerFidelity() {
  const isGFPGANAvailable = useAppSelector(
    (state: RootState) => state.system.isGFPGANAvailable
  );

  const codeformerFidelity = useAppSelector(
    (state: RootState) => state.postprocessing.codeformerFidelity
  );

  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  return (
    <IAISlider
      isDisabled={!isGFPGANAvailable}
      label={t('parameters.codeformerFidelity')}
      step={0.05}
      min={0}
      max={1}
      onChange={(v) => dispatch(setCodeformerFidelity(v))}
      handleReset={() => dispatch(setCodeformerFidelity(1))}
      value={codeformerFidelity}
      withReset
      withSliderMarks
      withInput
    />
  );
}
