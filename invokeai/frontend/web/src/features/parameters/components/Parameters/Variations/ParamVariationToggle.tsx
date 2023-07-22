import type { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISwitch from 'common/components/IAISwitch';
import { setShouldGenerateVariations } from 'features/parameters/store/generationSlice';
import { ChangeEvent } from 'react';

export const ParamVariationToggle = () => {
  const dispatch = useAppDispatch();

  const shouldGenerateVariations = useAppSelector(
    (state: RootState) => state.generation.shouldGenerateVariations
  );

  const handleChange = (e: ChangeEvent<HTMLInputElement>) =>
    dispatch(setShouldGenerateVariations(e.target.checked));

  return (
    <IAISwitch
      label="Enable Variations"
      isChecked={shouldGenerateVariations}
      onChange={handleChange}
    />
  );
};
