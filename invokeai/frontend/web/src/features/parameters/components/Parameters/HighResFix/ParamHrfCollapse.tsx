import { Flex } from '@chakra-ui/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { RootState, stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import IAICollapse from 'common/components/IAICollapse';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import ParamHrfMethod from './ParamHrfMethod';
import ParamHrfStrength from './ParamHrfStrength';
import ParamHrfToggle from './ParamHrfToggle';

const selector = createMemoizedSelector(stateSelector, (state: RootState) => {
  const { hrfEnabled } = state.generation;

  return { hrfEnabled };
});

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
