import { Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAICollapse from 'common/components/IAICollapse';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { ParamHiresStrength } from './ParamHiresStrength';
import { ParamHiresToggle } from './ParamHiresToggle';

const selector = createSelector(
  stateSelector,
  (state) => {
    const activeLabel = state.postprocessing.hiresFix ? 'Enabled' : undefined;

    return { activeLabel };
  },
  defaultSelectorOptions
);

const ParamHiresCollapse = () => {
  const { t } = useTranslation();
  const { activeLabel } = useAppSelector(selector);

  const isHiresEnabled = useFeatureStatus('hires').isFeatureEnabled;

  if (!isHiresEnabled) {
    return null;
  }

  return (
    <IAICollapse label={t('parameters.hiresOptim')} activeLabel={activeLabel}>
      <Flex sx={{ gap: 2, flexDirection: 'column' }}>
        <ParamHiresToggle />
        <ParamHiresStrength />
      </Flex>
    </IAICollapse>
  );
};

export default memo(ParamHiresCollapse);
