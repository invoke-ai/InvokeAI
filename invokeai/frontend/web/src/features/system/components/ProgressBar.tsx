import { Progress, Tooltip } from '@invoke-ai/ui-library';
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
  const message = useAppSelector((s) => s.system.denoiseProgress?.message);
  const hasSteps = useAppSelector((s) => Boolean(s.system.denoiseProgress?.percentage !== undefined));
  const value = useAppSelector(selectProgressValue);

  return (
    <Tooltip label={message} placement="end">
      <Progress
        value={value}
        aria-label={t('accessibility.invokeProgressBar')}
        isIndeterminate={isConnected && Boolean(queueStatus?.queue.in_progress) && !hasSteps}
        h={2}
        w="full"
        colorScheme="invokeBlue"
      />
    </Tooltip>
  );
};

export default memo(ProgressBar);
