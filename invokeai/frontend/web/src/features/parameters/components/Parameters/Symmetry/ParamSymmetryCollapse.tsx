import { Flex } from '@chakra-ui/react';
import { memo } from 'react';
import ParamSymmetryHorizontal from './ParamSymmetryHorizontal';
import ParamSymmetryVertical from './ParamSymmetryVertical';

import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAICollapse from 'common/components/IAICollapse';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { useTranslation } from 'react-i18next';
import ParamSymmetryToggle from './ParamSymmetryToggle';

const selector = createSelector(
  stateSelector,
  (state) => ({
    activeLabel: state.generation.shouldUseSymmetry ? 'Enabled' : undefined,
  }),
  defaultSelectorOptions
);

const ParamSymmetryCollapse = () => {
  const { t } = useTranslation();
  const { activeLabel } = useAppSelector(selector);

  const isSymmetryEnabled = useFeatureStatus('symmetry').isFeatureEnabled;

  if (!isSymmetryEnabled) {
    return null;
  }

  return (
    <IAICollapse label={t('parameters.symmetry')} activeLabel={activeLabel}>
      <Flex sx={{ gap: 2, flexDirection: 'column' }}>
        <ParamSymmetryToggle />
        <ParamSymmetryHorizontal />
        <ParamSymmetryVertical />
      </Flex>
    </IAICollapse>
  );
};

export default memo(ParamSymmetryCollapse);
