import { RAND_SEED_MAX, RAND_SEED_MIN } from 'app/constants';
import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import randomInt from 'common/util/randomInt';
import { setSeed } from 'features/parameters/store/generationSlice';
import { useTranslation } from 'react-i18next';
import { FaRandom } from 'react-icons/fa';

export default function ParamSeedShuffle() {
  const dispatch = useAppDispatch();
  const shouldRandomizeSeed = useAppSelector(
    (state: RootState) => state.generation.shouldRandomizeSeed
  );
  const { t } = useTranslation();

  const handleClickRandomizeSeed = () =>
    dispatch(setSeed(randomInt(RAND_SEED_MIN, RAND_SEED_MAX)));

  return (
    <IAIIconButton
      size="sm"
      isDisabled={shouldRandomizeSeed}
      aria-label={t('parameters.shuffle')}
      tooltip={t('parameters.shuffle')}
      onClick={handleClickRandomizeSeed}
      icon={<FaRandom />}
    />
  );
}
