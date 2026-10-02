import { SimpleGrid, Stat } from '@chakra-ui/react';
import { useTranslation } from 'react-i18next';

import { useQueueCounts } from './queueDataStore';

const QueueStatCard = ({ value, label, danger }: { value: number; label: string; danger?: boolean }) => (
  <Stat.Root size="sm" colorScheme={danger ? 'red' : undefined} gap="0">
    <Stat.Label fontSize="xs" order="2">
      {label}
    </Stat.Label>
    <Stat.ValueText order="1">{value}</Stat.ValueText>
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
