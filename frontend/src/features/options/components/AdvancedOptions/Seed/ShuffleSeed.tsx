import { Button } from '@chakra-ui/react';
import React from 'react';
import { NUMPY_RAND_MAX, NUMPY_RAND_MIN } from 'app/constants';
import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import randomInt from 'common/util/randomInt';
import { setSeed } from 'features/options/store/optionsSlice';
import { useTranslation } from 'react-i18next';

export default function ShuffleSeed() {
  const dispatch = useAppDispatch();
  const shouldRandomizeSeed = useAppSelector(
    (state: RootState) => state.options.shouldRandomizeSeed
  );
  const { t } = useTranslation();

  const handleClickRandomizeSeed = () =>
    dispatch(setSeed(randomInt(NUMPY_RAND_MIN, NUMPY_RAND_MAX)));

  return (
    <Button
      size={'sm'}
      isDisabled={shouldRandomizeSeed}
      onClick={handleClickRandomizeSeed}
      padding="0 1.5rem"
    >
      <p>{t('options:shuffle')}</p>
    </Button>
  );
}
