import { NUMPY_RAND_MAX, NUMPY_RAND_MIN } from 'app/constants';
import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAINumberInput from 'common/components/IAINumberInput';
import { setSeed } from 'features/parameters/store/generationSlice';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export default function ParamSeed() {
  const seed = useAppSelector((state: RootState) => state.generation.seed);
  const shouldRandomizeSeed = useAppSelector(
    (state: RootState) => state.generation.shouldRandomizeSeed
  );
  const shouldGenerateVariations = useAppSelector(
    (state: RootState) => state.generation.shouldGenerateVariations
  );

  const { t } = useTranslation();

  const dispatch = useAppDispatch();

  const handleChangeSeed = useCallback(
    (v: number) => dispatch(setSeed(v)),
    [dispatch]
  );

  return (
    <IAINumberInput
      label={t('parameters.seed')}
      step={1}
      precision={0}
      flexGrow={1}
      min={NUMPY_RAND_MIN}
      max={NUMPY_RAND_MAX}
      isDisabled={shouldRandomizeSeed}
      isInvalid={seed < 0 && shouldGenerateVariations}
      onChange={handleChangeSeed}
      value={seed}
    />
  );
}
