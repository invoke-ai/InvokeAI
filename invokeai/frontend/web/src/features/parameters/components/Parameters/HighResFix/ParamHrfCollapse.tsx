import { Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { RootState, stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAICollapse from 'common/components/IAICollapse';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import ParamHrfStrength from './ParamHrfStrength';
import ParamHrfToggle from './ParamHrfToggle';
import ParamHrfMethod from './ParamHrfMethod';

const selector = createSelector(
  stateSelector,
  (state: RootState) => {
    const { hrfEnabled } = state.generation;

    return { hrfEnabled };
  },
  defaultSelectorOptions
);

export default function ParamHrfCollapse() {
  const { t } = useTranslation();
  const isHRFFeatureEnabled = useFeatureStatus('hrf').isFeatureEnabled;
  const { hrfEnabled } = useAppSelector(selector);
  const activeLabel = useMemo(() => {
    if (hrfEnabled) {
      return t('common.on');
    }
  }, [t, hrfEnabled]);

  if (!isHRFFeatureEnabled) {
    return null;
  }

  return (
    <IAICollapse label={t('hrf.hrf')} activeLabel={activeLabel}>
      <Flex sx={{ flexDir: 'column', gap: 2 }}>
        <ParamHrfToggle />
        <ParamHrfStrength />
        <ParamHrfMethod />
      </Flex>
    </IAICollapse>
  );
}
