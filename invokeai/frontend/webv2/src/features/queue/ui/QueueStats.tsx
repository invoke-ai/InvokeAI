import { SimpleGrid, Stat } from '@chakra-ui/react';
import { useTranslation } from 'react-i18next';

import { useQueueCounts } from './queueDataStore';

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
  const { t } = useTranslation();
  const counts = useQueueCounts();

  return (
    <SimpleGrid columns={4}>
      <QueueStatCard label={t('common.done')} value={counts.completed} />
      <QueueStatCard danger={counts.failed > 0} label={t('common.failed')} value={counts.failed} />
      <QueueStatCard label={t('common.canceled')} value={counts.canceled} />
      <QueueStatCard label={t('common.total')} value={counts.total} />
    </SimpleGrid>
  );
};
