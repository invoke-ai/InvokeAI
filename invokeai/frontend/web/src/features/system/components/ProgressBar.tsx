import { Progress } from '@chakra-ui/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetQueueStatusQuery } from 'services/api/endpoints/queue';
const progressBarSelector = createMemoizedSelector(
  stateSelector,
  ({ system }) => {
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
      h="full"
      w="full"
      borderRadius={2}
      colorScheme="accent"
    />
  );
};

export default memo(ProgressBar);
