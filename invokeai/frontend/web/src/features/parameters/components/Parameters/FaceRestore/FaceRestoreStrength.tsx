import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISlider from 'common/components/IAISlider';
import { setFacetoolStrength } from 'features/parameters/store/postprocessingSlice';
import { useTranslation } from 'react-i18next';

export default function FaceRestoreStrength() {
  const isGFPGANAvailable = useAppSelector(
    (state: RootState) => state.system.isGFPGANAvailable
  );

  const facetoolStrength = useAppSelector(
    (state: RootState) => state.postprocessing.facetoolStrength
  );

  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  return (
    <IAISlider
      isDisabled={!isGFPGANAvailable}
      label={t('parameters.strength')}
      step={0.05}
      min={0}
      max={1}
      onChange={(v) => dispatch(setFacetoolStrength(v))}
      handleReset={() => dispatch(setFacetoolStrength(0.75))}
      value={facetoolStrength}
      withReset
      withSliderMarks
      withInput
    />
  );
}
