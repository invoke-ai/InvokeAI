import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetNextQueueItemQuery } from 'services/api/endpoints/queue';
import QueueItemCard from './common/QueueItemCard';

const NextQueueItemCard = () => {
  const { t } = useTranslation();
  const { data: nextQueueItemData } = useGetNextQueueItemQuery();

  return (
    <QueueItemCard
      label={t('queue.next')}
      session_queue_item={nextQueueItemData}
    />
  );
};

export default memo(NextQueueItemCard);
