import React, { ChangeEvent } from 'react';
import {
  RootState,
  useAppDispatch,
  useAppSelector,
} from 'app/store';
import IAISwitch from 'common/components/IAISwitch';
import { setShouldGenerateVariations } from 'features/options/store/optionsSlice';

export default function GenerateVariationsToggle() {
  const shouldGenerateVariations = useAppSelector(
    (state: RootState) => state.options.shouldGenerateVariations
  );

  const dispatch = useAppDispatch();

  const handleChangeShouldGenerateVariations = (
    e: ChangeEvent<HTMLInputElement>
  ) => dispatch(setShouldGenerateVariations(e.target.checked));

  return (
    <IAISwitch
      isChecked={shouldGenerateVariations}
      width={'auto'}
      onChange={handleChangeShouldGenerateVariations}
    />
  );
}
