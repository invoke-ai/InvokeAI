import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetQueueStatusQuery } from 'services/api/endpoints/queue';

import StatusStatGroup from './common/StatusStatGroup';
import StatusStatItem from './common/StatusStatItem';

const QueueStatus = () => {
  const { data: queueStatus } = useGetQueueStatusQuery();
  const { t } = useTranslation();
  return (
    <StatusStatGroup data-testid="queue-status">
      <StatusStatItem label={t('queue.in_progress')} value={queueStatus?.queue.in_progress ?? 0} />
      <StatusStatItem label={t('queue.pending')} value={queueStatus?.queue.pending ?? 0} />
      <StatusStatItem label={t('queue.completed')} value={queueStatus?.queue.completed ?? 0} />
      <StatusStatItem label={t('queue.failed')} value={queueStatus?.queue.failed ?? 0} />
      <StatusStatItem label={t('queue.canceled')} value={queueStatus?.queue.canceled ?? 0} />
      <StatusStatItem label={t('queue.total')} value={queueStatus?.queue.total ?? 0} />
    </StatusStatGroup>
  );
};

export default memo(QueueStatus);
