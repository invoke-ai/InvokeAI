import { useTranslation } from 'react-i18next';
import { Box, Flex } from '@chakra-ui/react';
import IAICollapse from 'common/components/IAICollapse';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { setSeamless } from 'features/parameters/store/generationSlice';
import { memo } from 'react';
import { createSelector } from '@reduxjs/toolkit';
import { generationSelector } from 'features/parameters/store/generationSelectors';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import ParamSeamlessXAxis from './ParamSeamlessXAxis';
import ParamSeamlessYAxis from './ParamSeamlessYAxis';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';

const selector = createSelector(
  generationSelector,
  (generation) => {
    const { shouldUseSeamless, seamlessXAxis, seamlessYAxis } = generation;

    return { shouldUseSeamless, seamlessXAxis, seamlessYAxis };
  },
  defaultSelectorOptions
);

const ParamSeamlessCollapse = () => {
  const { t } = useTranslation();
  const { shouldUseSeamless } = useAppSelector(selector);

  const isSeamlessEnabled = useFeatureStatus('seamless').isFeatureEnabled;

  const dispatch = useAppDispatch();

  const handleToggle = () => dispatch(setSeamless(!shouldUseSeamless));

  if (!isSeamlessEnabled) {
    return null;
  }

  return (
    <IAICollapse
      label={t('parameters.seamlessTiling')}
      isOpen={shouldUseSeamless}
      onToggle={handleToggle}
      withSwitch
    >
      <Flex sx={{ gap: 5 }}>
        <Box flexGrow={1}>
          <ParamSeamlessXAxis />
        </Box>
        <Box flexGrow={1}>
          <ParamSeamlessYAxis />
        </Box>
      </Flex>
    </IAICollapse>
  );
};

export default memo(ParamSeamlessCollapse);
