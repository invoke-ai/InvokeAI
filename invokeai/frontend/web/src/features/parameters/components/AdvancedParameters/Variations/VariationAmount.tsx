import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAISlider from 'common/components/IAISlider';
import { setVariationAmount } from 'features/parameters/store/generationSlice';
import { useTranslation } from 'react-i18next';

export default function VariationAmount() {
  const variationAmount = useAppSelector(
    (state: RootState) => state.generation.variationAmount
  );

  const shouldGenerateVariations = useAppSelector(
    (state: RootState) => state.generation.shouldGenerateVariations
  );

  const { t } = useTranslation();

  const dispatch = useAppDispatch();

  return (
    <IAISlider
      label={t('parameters.variationAmount')}
      value={variationAmount}
      step={0.01}
      min={0}
      max={1}
      isSliderDisabled={!shouldGenerateVariations}
      isInputDisabled={!shouldGenerateVariations}
      isResetDisabled={!shouldGenerateVariations}
      onChange={(v) => dispatch(setVariationAmount(v))}
      handleReset={() => dispatch(setVariationAmount(0.1))}
      withInput
      withReset
      withSliderMarks
    />
  );
}
