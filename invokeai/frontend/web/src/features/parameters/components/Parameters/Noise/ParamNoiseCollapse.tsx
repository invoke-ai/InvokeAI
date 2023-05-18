import { useTranslation } from 'react-i18next';
import { Flex } from '@chakra-ui/react';
import IAICollapse from 'common/components/IAICollapse';
import ParamPerlinNoise from './ParamPerlinNoise';
import ParamNoiseThreshold from './ParamNoiseThreshold';
import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { setShouldUseNoiseSettings } from 'features/parameters/store/generationSlice';
import { memo } from 'react';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';

const ParamNoiseCollapse = () => {
  const { t } = useTranslation();

  const isNoiseEnabled = useFeatureStatus('noise').isFeatureEnabled;

  const shouldUseNoiseSettings = useAppSelector(
    (state: RootState) => state.generation.shouldUseNoiseSettings
  );

  const dispatch = useAppDispatch();

  const handleToggle = () =>
    dispatch(setShouldUseNoiseSettings(!shouldUseNoiseSettings));

  if (!isNoiseEnabled) {
    return null;
  }

  return (
    <IAICollapse
      label={t('parameters.noiseSettings')}
      isOpen={shouldUseNoiseSettings}
      onToggle={handleToggle}
      withSwitch
    >
      <Flex sx={{ gap: 2, flexDirection: 'column' }}>
        <ParamPerlinNoise />
        <ParamNoiseThreshold />
      </Flex>
    </IAICollapse>
  );
};

export default memo(ParamNoiseCollapse);
