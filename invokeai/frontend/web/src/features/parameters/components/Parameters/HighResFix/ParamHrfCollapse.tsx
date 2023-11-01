import { Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { RootState, stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAICollapse from 'common/components/IAICollapse';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import ParamHrfHeight from './ParamHrfHeight';
import ParamHrfStrength from './ParamHrfStrength';
import ParamHrfToggle from './ParamHrfToggle';
import ParamHrfWidth from './ParamHrfWidth';

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
    <IAICollapse label="High Resolution Fix" activeLabel={activeLabel}>
      <Flex sx={{ flexDir: 'column', gap: 2 }}>
        <ParamHrfToggle />
        {hrfEnabled && (
          <Flex
            sx={{
              gap: 2,
              p: 4,
              borderRadius: 4,
              flexDirection: 'column',
              w: 'full',
              bg: 'base.100',
              _dark: {
                bg: 'base.750',
              },
            }}
          >
            <ParamHrfWidth />
            <ParamHrfHeight />
          </Flex>
        )}
        {hrfEnabled && <ParamHrfStrength />}
      </Flex>
    </IAICollapse>
  );
}
