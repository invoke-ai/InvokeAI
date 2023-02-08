import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAINumberInput from 'common/components/IAINumberInput';
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
  const handleChangevariationAmount = (v: number) =>
    dispatch(setVariationAmount(v));

  return (
    <IAINumberInput
      label={t('parameters:variationAmount')}
      value={variationAmount}
      step={0.01}
      min={0}
      max={1}
      isDisabled={!shouldGenerateVariations}
      onChange={handleChangevariationAmount}
      isInteger={false}
    />
  );
}
