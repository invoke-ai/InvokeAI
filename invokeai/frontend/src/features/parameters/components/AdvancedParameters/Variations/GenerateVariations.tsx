import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAISwitch from 'common/components/IAISwitch';
import { setShouldGenerateVariations } from 'features/parameters/store/generationSlice';
import { ChangeEvent } from 'react';

export default function GenerateVariationsToggle() {
  const shouldGenerateVariations = useAppSelector(
    (state: RootState) => state.generation.shouldGenerateVariations
  );

  const dispatch = useAppDispatch();

  const handleChangeShouldGenerateVariations = (
    e: ChangeEvent<HTMLInputElement>
  ) => dispatch(setShouldGenerateVariations(e.target.checked));

  return (
    <IAISwitch
      isChecked={shouldGenerateVariations}
      width="auto"
      onChange={handleChangeShouldGenerateVariations}
    />
  );
}
