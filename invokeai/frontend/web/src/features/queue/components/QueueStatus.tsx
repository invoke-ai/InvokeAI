import { Stat, StatGroup, StatLabel, StatNumber } from '@chakra-ui/react';
import { memo } from 'react';
import { useGetQueueStatusQuery } from 'services/api/endpoints/queue';

const QueueStatus = () => {
  const { data: queueStatus } = useGetQueueStatusQuery();

  return (
    <StatGroup alignItems="center" justifyContent="center" w="full" h="full">
      <Stat w={24}>
        <StatLabel>In Progress</StatLabel>
        <StatNumber>{queueStatus?.queue.in_progress ?? 0}</StatNumber>
      </Stat>
      <Stat w={24}>
        <StatLabel>Pending</StatLabel>
        <StatNumber>{queueStatus?.queue.pending ?? 0}</StatNumber>
      </Stat>
      <Stat w={24}>
        <StatLabel>Completed</StatLabel>
        <StatNumber>{queueStatus?.queue.completed ?? 0}</StatNumber>
      </Stat>
      <Stat w={24}>
        <StatLabel>Failed</StatLabel>
        <StatNumber>{queueStatus?.queue.failed ?? 0}</StatNumber>
      </Stat>
      <Stat w={24}>
        <StatLabel>Canceled</StatLabel>
        <StatNumber>{queueStatus?.queue.canceled ?? 0}</StatNumber>
      </Stat>
      <Stat w={24}>
        <StatLabel>Total</StatLabel>
        <StatNumber>{queueStatus?.queue.total}</StatNumber>
      </Stat>
    </StatGroup>
  );
};

export default memo(QueueStatus);
