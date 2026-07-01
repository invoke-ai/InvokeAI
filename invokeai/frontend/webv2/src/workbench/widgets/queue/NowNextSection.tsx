import { Stack } from '@chakra-ui/react';
import { useTranslation } from 'react-i18next';

import { QueueItemRow } from './QueueItemRow';
import { useScopedCurrentBatchItems } from './queueScope';
import { SectionHeader } from './SectionHeader';

/** CURRENT BATCH — the running item plus every pending item in the same backend batch. */
export const CurrentBatchSection = () => {
  const { t } = useTranslation();
  const items = useScopedCurrentBatchItems();
  const count = items.length;

  if (count === 0) {
    return null;
  }

  return (
    <Stack gap="2">
      <SectionHeader count={count} title={t('widgets.queue.currentBatch')} />
      <Stack gap="1">
        {items.map((item) => (
          <QueueItemRow key={item.item_id} item={item} />
        ))}
      </Stack>
    </Stack>
  );
};
