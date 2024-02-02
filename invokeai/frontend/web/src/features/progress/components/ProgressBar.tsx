import { Progress } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetQueueStatusQuery } from 'services/api/endpoints/queue';

const ProgressBar = () => {
  const { t } = useTranslation();
  const { data: queueStatus } = useGetQueueStatusQuery();
  const isConnected = useAppSelector((s) => s.system.isConnected);
  const hasSteps = useAppSelector((s) => Boolean(s.progress.latestDenoiseProgress));
  const value = useAppSelector((s) => (s.progress.latestDenoiseProgress?.percentage ?? 0) * 100);
  const isIndeterminate = useMemo(() => {
    return isConnected && Boolean(queueStatus?.queue.in_progress) && !hasSteps;
  }, [hasSteps, isConnected, queueStatus?.queue.in_progress]);

  return (
    <Progress
      value={value}
      aria-label={t('accessibility.invokeProgressBar')}
      isIndeterminate={isIndeterminate}
      h={2}
      w="full"
      colorScheme="invokeBlue"
    />
  );
};

export default memo(ProgressBar);
