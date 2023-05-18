import { Flex } from '@chakra-ui/react';
import ParamSeed from './ParamSeed';
import { memo, useCallback } from 'react';
import ParamSeedShuffle from './ParamSeedShuffle';
import { useTranslation } from 'react-i18next';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { createSelector } from '@reduxjs/toolkit';
import { generationSelector } from 'features/parameters/store/generationSelectors';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { setShouldRandomizeSeed } from 'features/parameters/store/generationSlice';
import IAICollapse from 'common/components/IAICollapse';

const selector = createSelector(
  generationSelector,
  (generation) => {
    const { shouldRandomizeSeed } = generation;

    return { shouldRandomizeSeed };
  },
  defaultSelectorOptions
);

const ParamSeedSettings = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const { shouldRandomizeSeed } = useAppSelector(selector);

  const handleToggle = useCallback(
    () => dispatch(setShouldRandomizeSeed(!shouldRandomizeSeed)),
    [dispatch, shouldRandomizeSeed]
  );

  return (
    <IAICollapse
      label={t('parameters.seed')}
      isOpen={!shouldRandomizeSeed}
      onToggle={handleToggle}
      withSwitch
    >
      <Flex sx={{ gap: 4 }}>
        <ParamSeed />
        <ParamSeedShuffle />
      </Flex>
    </IAICollapse>
  );
};

export default memo(ParamSeedSettings);
