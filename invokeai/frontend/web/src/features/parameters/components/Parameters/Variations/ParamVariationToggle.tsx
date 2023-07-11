import type { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISwitch from 'common/components/IAISwitch';
import { setShouldGenerateVariations } from 'features/parameters/store/generationSlice';
import { ChangeEvent } from 'react';
import { useTranslation } from 'react-i18next';

export const ParamVariationToggle = () => {
  const dispatch = useAppDispatch();

  const shouldGenerateVariations = useAppSelector(
    (state: RootState) => state.generation.shouldGenerateVariations
  );

  const { t } = useTranslation();

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
