import { Box } from '@chakra-ui/react';
import { NUMPY_RAND_MAX, NUMPY_RAND_MIN } from 'app/constants';
import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIButton from 'common/components/IAIButton';
import randomInt from 'common/util/randomInt';
import { setSeed } from 'features/parameters/store/generationSlice';
import { useTranslation } from 'react-i18next';

export default function ParamSeedShuffle() {
  const dispatch = useAppDispatch();
  const shouldRandomizeSeed = useAppSelector(
    (state: RootState) => state.generation.shouldRandomizeSeed
  );
  const { t } = useTranslation();

  const handleClickRandomizeSeed = () =>
    dispatch(setSeed(randomInt(NUMPY_RAND_MIN, NUMPY_RAND_MAX)));

  return (
    <IAIButton
      size="sm"
      isDisabled={shouldRandomizeSeed}
      aria-label={t('parameters.shuffle')}
      tooltip={t('parameters.shuffle')}
      onClick={handleClickRandomizeSeed}
    >
      <Box px={2}> {t('parameters.shuffle')}</Box>
    </IAIButton>
  );
}
