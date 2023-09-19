import { Progress } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetQueueStatusQuery } from 'services/api/endpoints/queue';
const progressBarSelector = createSelector(
  stateSelector,
  ({ system }) => {
    return {
      hasSteps: Boolean(system.denoiseProgress),
      value: (system.denoiseProgress?.percentage ?? 0) * 100,
    };
  },
  defaultSelectorOptions
);

const ProgressBar = () => {
  const { t } = useTranslation();
  const { data: queueStatus } = useGetQueueStatusQuery();
  const { hasSteps, value } = useAppSelector(progressBarSelector);

  console.log(value);
  // const value = useMemo(() => {
  //   if (currentStep && Boolean(queueStatus?.queue.in_progress)) {
  //     return Math.round((currentStep * 100) / totalSteps);
  //   }
  //   return 0;
  // }, [currentStep, queueStatus?.queue.in_progress, totalSteps]);

  return (
    <Progress
      value={value}
      aria-label={t('accessibility.invokeProgressBar')}
      isIndeterminate={Boolean(queueStatus?.queue.in_progress) && !hasSteps}
      h="full"
      w="full"
      borderRadius={2}
      colorScheme="accent"
    />
  );
};

export default memo(ProgressBar);
