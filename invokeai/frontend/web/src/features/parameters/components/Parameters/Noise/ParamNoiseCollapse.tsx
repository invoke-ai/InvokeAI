import { Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAICollapse from 'common/components/IAICollapse';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { ParamCpuNoiseToggle } from './ParamCpuNoise';
import ParamNoiseThreshold from './ParamNoiseThreshold';
import { ParamNoiseToggle } from './ParamNoiseToggle';
import ParamPerlinNoise from './ParamPerlinNoise';

const selector = createSelector(
  stateSelector,
  (state) => {
    const { shouldUseNoiseSettings } = state.generation;
    return {
      activeLabel: shouldUseNoiseSettings ? 'Enabled' : undefined,
    };
  },
  defaultSelectorOptions
);

const ParamNoiseCollapse = () => {
  const { t } = useTranslation();

  const isNoiseEnabled = useFeatureStatus('noise').isFeatureEnabled;
  const isPerlinNoiseEnabled = useFeatureStatus('perlinNoise').isFeatureEnabled;
  const isNoiseThresholdEnabled =
    useFeatureStatus('noiseThreshold').isFeatureEnabled;

  const { activeLabel } = useAppSelector(selector);

  if (!isNoiseEnabled) {
    return null;
  }

  return (
    <IAICollapse
      label={t('parameters.noiseSettings')}
      activeLabel={activeLabel}
    >
      <Flex sx={{ gap: 2, flexDirection: 'column' }}>
        <ParamNoiseToggle />
        <ParamCpuNoiseToggle />
        {isPerlinNoiseEnabled && <ParamPerlinNoise />}
        {isNoiseThresholdEnabled && <ParamNoiseThreshold />}
      </Flex>
    </IAICollapse>
  );
};

export default memo(ParamNoiseCollapse);
