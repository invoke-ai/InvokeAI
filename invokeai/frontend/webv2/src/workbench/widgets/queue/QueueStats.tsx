import { SimpleGrid, Stat } from '@chakra-ui/react';

import { useScopedQueueCounts } from './queueScope';

/**
 * The four headline counts (done / failed / canceled / total) as a row of cards.
 * Numbers are tabular so they stay aligned as the queue churns; the failed count
 * turns danger-colored only when there's something to call out.
 */
const QueueStatCard = ({ value, label, danger }: { value: number; label: string; danger?: boolean }) => (
  <Stat.Root size="sm" colorScheme={danger ? 'red' : undefined} gap="0">
    <Stat.ValueText>{value}</Stat.ValueText>
    <Stat.Label fontSize="xs">{label}</Stat.Label>
  </Stat.Root>
);

export const QueueStats = () => {
  const counts = useScopedQueueCounts();

  return (
    <SimpleGrid columns={4}>
      <QueueStatCard label="done" value={counts.completed} />
      <QueueStatCard danger={counts.failed > 0} label="failed" value={counts.failed} />
      <QueueStatCard label="canceled" value={counts.canceled} />
      <QueueStatCard label="total" value={counts.total} />
    </SimpleGrid>
  );
};
