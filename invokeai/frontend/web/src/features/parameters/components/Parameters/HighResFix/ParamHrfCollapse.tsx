import { Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { RootState, stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAICollapse from 'common/components/IAICollapse';
import { useMemo } from 'react';
import ParamHrfStrength from './ParamHrfStrength';
import ParamHrfToggle from './ParamHrfToggle';
import ParamHrfMethod from './ParamHrfMethod';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';

const selector = createSelector(
  stateSelector,
  (state: RootState) => {
    const { hrfEnabled } = state.generation;

    return { hrfEnabled };
  },
  defaultSelectorOptions
);

export default function ParamHrfCollapse() {
  const isHRFFeatureEnabled = useFeatureStatus('hrf').isFeatureEnabled;
  const { hrfEnabled } = useAppSelector(selector);
  const activeLabel = useMemo(() => {
    if (hrfEnabled) {
      return 'On';
    } else {
      return 'Off';
    }
  }, [hrfEnabled]);

  if (!isHRFFeatureEnabled) {
    return null;
  }

  return (
    <IAICollapse label="High Resolution Fix" activeLabel={activeLabel}>
      <Flex sx={{ flexDir: 'column', gap: 2 }}>
        <ParamHrfToggle />
        {hrfEnabled && <ParamHrfStrength />}
        {hrfEnabled && <ParamHrfMethod />}
      </Flex>
    </IAICollapse>
  );
}
