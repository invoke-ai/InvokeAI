import { Box, Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAICollapse from 'common/components/IAICollapse';
import { generationSelector } from 'features/parameters/store/generationSelectors';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import ParamSeamlessXAxis from './ParamSeamlessXAxis';
import ParamSeamlessYAxis from './ParamSeamlessYAxis';

const getActiveLabel = (seamlessXAxis: boolean, seamlessYAxis: boolean) => {
  if (seamlessXAxis && seamlessYAxis) {
    return 'X & Y';
  }

  if (seamlessXAxis) {
    return 'X';
  }

  if (seamlessYAxis) {
    return 'Y';
  }
};

const selector = createSelector(
  generationSelector,
  (generation) => {
    const { seamlessXAxis, seamlessYAxis } = generation;

    const activeLabel = getActiveLabel(seamlessXAxis, seamlessYAxis);
    return { activeLabel };
  },
  defaultSelectorOptions
);

const ParamSeamlessCollapse = () => {
  const { t } = useTranslation();
  const { activeLabel } = useAppSelector(selector);

  const isSeamlessEnabled = useFeatureStatus('seamless').isFeatureEnabled;

  if (!isSeamlessEnabled) {
    return null;
  }

  return (
    <IAICollapse
      label={t('parameters.seamlessTiling')}
      activeLabel={activeLabel}
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
