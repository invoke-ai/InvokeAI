import { Progress } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { useIsQueueStarted } from 'features/queue/hooks/useIsQueueStarted';
import { SystemState } from 'features/system/store/systemSlice';
import { isEqual } from 'lodash-es';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { systemSelector } from '../store/systemSelectors';

const progressBarSelector = createSelector(
  systemSelector,
  (system: SystemState) => {
    return {
      currentStep: system.currentStep,
      totalSteps: system.totalSteps,
      currentStatusHasSteps: system.currentStatusHasSteps,
    };
  },
  {
    memoizeOptions: { resultEqualityCheck: isEqual },
  }
);

const ProgressBar = () => {
  const { t } = useTranslation();
  const isStarted = useIsQueueStarted();
  const { currentStep, totalSteps, currentStatusHasSteps } =
    useAppSelector(progressBarSelector);

  const value = useMemo(() => {
    if (currentStep && isStarted) {
      return Math.round((currentStep * 100) / totalSteps);
    }
    return 0;
  }, [currentStep, isStarted, totalSteps]);

  return (
    <Progress
      value={value}
      aria-label={t('accessibility.invokeProgressBar')}
      isIndeterminate={isStarted && !currentStatusHasSteps}
      h="full"
      w="full"
      borderRadius={2}
      colorScheme="accent"
    />
  );
};

export default memo(ProgressBar);
