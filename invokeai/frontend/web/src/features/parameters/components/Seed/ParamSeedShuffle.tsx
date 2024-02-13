import { Button } from '@invoke-ai/ui-library';
import { NUMPY_RAND_MAX, NUMPY_RAND_MIN } from 'app/constants';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import randomInt from 'common/util/randomInt';
import { setSeed } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiShuffleBold } from 'react-icons/pi';

export const ParamSeedShuffle = memo(() => {
  const dispatch = useAppDispatch();
  const shouldRandomizeSeed = useAppSelector((s) => s.generation.shouldRandomizeSeed);
  const { t } = useTranslation();

  const handleClickRandomizeSeed = useCallback(
    () => dispatch(setSeed(randomInt(NUMPY_RAND_MIN, NUMPY_RAND_MAX))),
    [dispatch]
  );

  return (
    <Button
      size="sm"
      isDisabled={shouldRandomizeSeed}
      onClick={handleClickRandomizeSeed}
      leftIcon={<PiShuffleBold />}
      flexShrink={0}
    >
      {t('parameters.shuffle')}
    </Button>
  );
});

ParamSeedShuffle.displayName = 'ParamSeedShuffle';
