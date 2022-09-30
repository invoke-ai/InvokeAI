import React from 'react';
import {
  RootState,
  useAppDispatch,
  useAppSelector,
} from '../../../../app/store';
import SDNumberInput from '../../../../common/components/SDNumberInput';
import { setVariationAmount } from '../../optionsSlice';

export default function VariationAmount() {
  const variationAmount = useAppSelector(
    (state: RootState) => state.options.variationAmount
  );

  const shouldGenerateVariations = useAppSelector(
    (state: RootState) => state.options.shouldGenerateVariations
  );

  const dispatch = useAppDispatch();
  const handleChangevariationAmount = (v: number) =>
    dispatch(setVariationAmount(v));

  return (
    <SDNumberInput
      label="Variation Amount"
      value={variationAmount}
      step={0.01}
      min={0}
      max={1}
      isDisabled={!shouldGenerateVariations}
      onChange={handleChangevariationAmount}
      width="90px"
      isInteger={false}
    />
  );
}
