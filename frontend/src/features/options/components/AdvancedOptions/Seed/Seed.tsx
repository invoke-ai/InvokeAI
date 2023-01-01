import React from 'react';
import { NUMPY_RAND_MAX, NUMPY_RAND_MIN } from 'app/constants';
import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAINumberInput from 'common/components/IAINumberInput';
import { setSeed } from 'features/options/store/optionsSlice';
import { useTranslation } from 'react-i18next';

export default function Seed() {
  const seed = useAppSelector((state: RootState) => state.options.seed);
  const shouldRandomizeSeed = useAppSelector(
    (state: RootState) => state.options.shouldRandomizeSeed
  );
  const shouldGenerateVariations = useAppSelector(
    (state: RootState) => state.options.shouldGenerateVariations
  );

  const { t } = useTranslation();

  const dispatch = useAppDispatch();

  const handleChangeSeed = (v: number) => dispatch(setSeed(v));

  return (
    <IAINumberInput
      label={t('options:seed')}
      step={1}
      precision={0}
      flexGrow={1}
      min={NUMPY_RAND_MIN}
      max={NUMPY_RAND_MAX}
      isDisabled={shouldRandomizeSeed}
      isInvalid={seed < 0 && shouldGenerateVariations}
      onChange={handleChangeSeed}
      value={seed}
      width="auto"
    />
  );
}
