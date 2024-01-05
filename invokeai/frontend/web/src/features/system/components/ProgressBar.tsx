import { Progress } from '@chakra-ui/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { selectSystemSlice } from 'features/system/store/systemSlice';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetQueueStatusQuery } from 'services/api/endpoints/queue';
const progressBarSelector = createMemoizedSelector(
  selectSystemSlice,
  (system) => {
    return {
      isConnected: system.isConnected,
      hasSteps: Boolean(system.denoiseProgress),
      value: (system.denoiseProgress?.percentage ?? 0) * 100,
    };
  }
);

const ProgressBar = () => {
  const { t } = useTranslation();
  const { data: queueStatus } = useGetQueueStatusQuery();
  const { hasSteps, value, isConnected } = useAppSelector(progressBarSelector);

  return (
    <Progress
      value={value}
      aria-label={t('accessibility.invokeProgressBar')}
      isIndeterminate={
        isConnected && Boolean(queueStatus?.queue.in_progress) && !hasSteps
      }
      h={2}
      w="full"
      colorScheme="blue"
    />
  );
};

export default memo(ProgressBar);
