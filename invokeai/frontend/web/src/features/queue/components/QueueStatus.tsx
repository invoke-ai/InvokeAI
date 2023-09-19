import { Stat, StatGroup, StatLabel, StatNumber } from '@chakra-ui/react';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetQueueStatusQuery } from 'services/api/endpoints/queue';

const QueueStatus = () => {
  const { data: queueStatus } = useGetQueueStatusQuery();
  const { t } = useTranslation();
  return (
    <StatGroup alignItems="center" justifyContent="center" w="full" h="full">
      <Stat w={24}>
        <StatLabel>{t('queue.in_progress')}</StatLabel>
        <StatNumber>{queueStatus?.queue.in_progress ?? 0}</StatNumber>
      </Stat>
      <Stat w={24}>
        <StatLabel>{t('queue.pending')}</StatLabel>
        <StatNumber>{queueStatus?.queue.pending ?? 0}</StatNumber>
      </Stat>
      <Stat w={24}>
        <StatLabel>{t('queue.completed')}</StatLabel>
        <StatNumber>{queueStatus?.queue.completed ?? 0}</StatNumber>
      </Stat>
      <Stat w={24}>
        <StatLabel>{t('queue.failed')}</StatLabel>
        <StatNumber>{queueStatus?.queue.failed ?? 0}</StatNumber>
      </Stat>
      <Stat w={24}>
        <StatLabel>{t('queue.canceled')}</StatLabel>
        <StatNumber>{queueStatus?.queue.canceled ?? 0}</StatNumber>
      </Stat>
      <Stat w={24}>
        <StatLabel>{t('queue.total')}</StatLabel>
        <StatNumber>{queueStatus?.queue.total}</StatNumber>
      </Stat>
    </StatGroup>
  );
};

export default memo(QueueStatus);
