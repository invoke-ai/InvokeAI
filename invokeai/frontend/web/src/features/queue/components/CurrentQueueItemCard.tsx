import { memo } from 'react';
import QueueItemCard from './common/QueueItemCard';
import { useGetCurrentQueueItemQuery } from 'services/api/endpoints/queue';
import { useTranslation } from 'react-i18next';

const CurrentQueueItemCard = () => {
  const { t } = useTranslation();
  const { data: currentQueueItemData } = useGetCurrentQueueItemQuery();

  return (
    <QueueItemCard
      label={t('queue.current')}
      session_queue_item={currentQueueItemData}
    />
  );
};

export default memo(CurrentQueueItemCard);
