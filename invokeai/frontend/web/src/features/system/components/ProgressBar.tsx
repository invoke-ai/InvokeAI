import { Progress } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectSystemSlice } from 'features/system/store/systemSlice';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetQueueStatusQuery } from 'services/api/endpoints/queue';

const selectProgressValue = createSelector(
  selectSystemSlice,
  (system) => (system.denoiseProgress?.percentage ?? 0) * 100
);

const ProgressBar = () => {
  const { t } = useTranslation();
  const { data: queueStatus } = useGetQueueStatusQuery();
  const isConnected = useAppSelector((s) => s.system.isConnected);
  const hasSteps = useAppSelector((s) => Boolean(s.system.denoiseProgress));
  const value = useAppSelector(selectProgressValue);

  return (
    <Progress
      value={value}
      aria-label={t('accessibility.invokeProgressBar')}
      isIndeterminate={isConnected && Boolean(queueStatus?.queue.in_progress) && !hasSteps}
      h={2}
      w="full"
      colorScheme="invokeBlue"
    />
  );
};

export default memo(ProgressBar);
