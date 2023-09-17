import { Progress } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { SystemState } from 'features/system/store/systemSlice';
import { isEqual } from 'lodash-es';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetQueueStatusQuery } from 'services/api/endpoints/queue';
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
  const { data: queueStatus } = useGetQueueStatusQuery();
  const { currentStep, totalSteps, currentStatusHasSteps } =
    useAppSelector(progressBarSelector);

  const value = useMemo(() => {
    if (currentStep && Boolean(queueStatus?.queue.in_progress)) {
      return Math.round((currentStep * 100) / totalSteps);
    }
    return 0;
  }, [currentStep, queueStatus?.queue.in_progress, totalSteps]);

  return (
    <Progress
      value={value}
      aria-label={t('accessibility.invokeProgressBar')}
      isIndeterminate={
        Boolean(queueStatus?.queue.in_progress) && !currentStatusHasSteps
      }
      h="full"
      w="full"
      borderRadius={2}
      colorScheme="accent"
    />
  );
};

export default memo(ProgressBar);
