import { RAND_SEED_MAX, RAND_SEED_MIN } from 'app/constants';
import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAINumberInput from 'common/components/IAINumberInput';
import { setSeed } from 'features/parameters/store/generationSlice';
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

  const handleChangeSeed = (v: number) => dispatch(setSeed(v));

  return (
    <IAINumberInput
      label={t('parameters.seed')}
      step={1}
      precision={0}
      flexGrow={1}
      min={RAND_SEED_MIN}
      max={RAND_SEED_MAX}
      isDisabled={shouldRandomizeSeed}
      isInvalid={seed < 0 && shouldGenerateVariations}
      onChange={handleChangeSeed}
      value={seed}
    />
  );
}
