import { Button } from '@invoke-ai/ui-library';
import { NUMPY_RAND_MAX, NUMPY_RAND_MIN } from 'app/constants';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import randomInt from 'common/util/randomInt';
import {
  selectImagen3EnhancePrompt,
  selectIsImagen3,
  selectShouldRandomizeSeed,
  setSeed,
} from 'features/controlLayers/store/paramsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiShuffleBold } from 'react-icons/pi';

export const ParamSeedShuffle = memo(() => {
  const dispatch = useAppDispatch();
  const shouldRandomizeSeed = useAppSelector(selectShouldRandomizeSeed);
  const isImagen3 = useAppSelector(selectIsImagen3);
  const imagen3EnhancePrompt = useAppSelector(selectImagen3EnhancePrompt);

  const { t } = useTranslation();

  const handleClickRandomizeSeed = useCallback(
    () => dispatch(setSeed(randomInt(NUMPY_RAND_MIN, NUMPY_RAND_MAX))),
    [dispatch]
  );

  return (
    <Button
      size="sm"
      isDisabled={shouldRandomizeSeed || (isImagen3 && imagen3EnhancePrompt)}
      onClick={handleClickRandomizeSeed}
      leftIcon={<PiShuffleBold />}
      flexShrink={0}
    >
      {t('parameters.shuffle')}
    </Button>
  );
});

ParamSeedShuffle.displayName = 'ParamSeedShuffle';
