import { Progress } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { SystemState } from 'features/system/store/systemSlice';
import { isEqual } from 'lodash-es';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { systemSelector } from '../store/systemSelectors';
import { useGetQueueStatusQuery } from 'services/api/endpoints/queue';

const progressBarSelector = createSelector(
  systemSelector,
  (system: SystemState) => {
    return {
      isProcessing: system.isProcessing,
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
  const { isProcessing } = useGetQueueStatusQuery(undefined, {
    selectFromResult: ({ data }) => {
      if (!data) {
        return { isProcessing: false };
      }

      return { isProcessing: data.started };
    },
  });
  const { currentStep, totalSteps, currentStatusHasSteps } =
    useAppSelector(progressBarSelector);

  const value = useMemo(() => {
    if (currentStep && isProcessing) {
      return Math.round((currentStep * 100) / totalSteps);
    }
    return 0;
  }, [currentStep, isProcessing, totalSteps]);

  return (
    <Progress
      value={value}
      aria-label={t('accessibility.invokeProgressBar')}
      isIndeterminate={isProcessing && !currentStatusHasSteps}
      h="full"
      w="full"
      borderRadius={2}
      colorScheme="accent"
    />
  );
};

export default memo(ProgressBar);
