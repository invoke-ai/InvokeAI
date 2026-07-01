import { Stack, Text } from '@chakra-ui/react';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import type { QueueFilterId } from './queueFilters';

import { useQueueLoadState } from './queueDataStore';
import { matchesFilter } from './queueFilters';
import { QueueItemRow } from './QueueItemRow';
import { useScopedCurrentBatchItems, useScopedRecentItems } from './queueScope';
import { SectionHeader } from './SectionHeader';

/**
 * RECENT — the windowed queue history, filtered by the active tab. The running
 * and next items are excluded here since NOW & NEXT already shows them.
 */
export const RecentSection = ({ filter }: { filter: QueueFilterId }) => {
  const { t } = useTranslation();
  const currentBatchItems = useScopedCurrentBatchItems();
  const items = useScopedRecentItems();
  const { error, loadState } = useQueueLoadState();

  const excluded = useMemo(() => new Set(currentBatchItems.map((item) => item.item_id)), [currentBatchItems]);
  const filtered = useMemo(
    () => items.filter((item) => !excluded.has(item.item_id) && matchesFilter(item.status, filter)),
    [excluded, filter, items]
  );

  return (
    <Stack gap="2">
      <SectionHeader count={filtered.length} title={t('common.recent')} />
      {filtered.length === 0 ? (
        <Text color={loadState === 'error' ? 'fg.error' : 'fg.subtle'} fontSize="2xs" px="1">
          {loadState === 'loading'
            ? t('widgets.queue.loading')
            : loadState === 'error'
              ? error
              : t('common.nothingHereYet')}
        </Text>
      ) : (
        <Stack gap="1">
          {filtered.map((item) => (
            <QueueItemRow key={item.item_id} item={item} />
          ))}
        </Stack>
      )}
    </Stack>
  );
};
