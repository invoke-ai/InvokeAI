import { Stack } from '@chakra-ui/react';
import { useTranslation } from 'react-i18next';

import { useCurrentBatchItems } from './queueDataStore';
import { QueueItemRow } from './QueueItemRow';
import { SectionHeader } from './SectionHeader';

/** CURRENT BATCH — the running item plus every pending item in the same backend batch. */
export const CurrentBatchSection = () => {
  const { t } = useTranslation();
  const items = useCurrentBatchItems();
  const count = items.length;

  if (count === 0) {
    return null;
  }

  return (
    <Stack gap="2">
      <SectionHeader count={count} title={t('widgets.queue.currentBatch')} />
      <Stack gap="1">
        {items.map((item) => (
          <QueueItemRow key={item.id} item={item} />
        ))}
      </Stack>
    </Stack>
  );
};
