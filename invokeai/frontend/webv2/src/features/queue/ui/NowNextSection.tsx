import { Stack } from '@chakra-ui/react';
import { ListSectionHeader } from '@platform/ui/list/ListSectionHeader';
import { ListStack } from '@platform/ui/list/ListStack';
import { useTranslation } from 'react-i18next';

import type { QueueItemRevealRequest } from './queueUiStore';

import { useCurrentBatchItems } from './queueDataStore';
import { QueueItemRow } from './QueueItemRow';

/** CURRENT BATCH — the running item plus every pending item in the same backend batch. */
export const CurrentBatchSection = ({ revealRequest = null }: { revealRequest?: QueueItemRevealRequest | null }) => {
  const { t } = useTranslation();
  const items = useCurrentBatchItems();
  const count = items.length;

  if (count === 0) {
    return null;
  }

  return (
    // Row surfaces bleed into the widget padding so their content lines up with the controls above.
    <Stack gap="1" mx="-2">
      <ListSectionHeader count={count} label={t('widgets.queue.currentBatch')} />
      <ListStack dividers label={t('widgets.queue.currentBatch')}>
        {items.map((item) => (
          <QueueItemRow
            key={item.id}
            item={item}
            revealRequest={item.id === revealRequest?.itemId ? revealRequest : null}
          />
        ))}
      </ListStack>
    </Stack>
  );
};
