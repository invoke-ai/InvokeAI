import { Stack, Text } from '@chakra-ui/react';
import { useEffect, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import type { QueueFilterId } from './queueFilters';

import { useCurrentBatchItems, useQueueLoadState, useRecentItems } from './queueDataStore';
import { matchesFilter } from './queueFilters';
import { QueueItemRow } from './QueueItemRow';
import { clearPendingQueueItemReveal } from './queueUiStore';
import { SectionHeader } from './SectionHeader';

/**
 * RECENT — the windowed queue history, filtered by the active tab. The running
 * and next items are excluded here since NOW & NEXT already shows them.
 */
export const RecentSection = ({
  filter,
  revealItemId = null,
}: {
  filter: QueueFilterId;
  revealItemId?: number | null;
}) => {
  const { t } = useTranslation();
  const currentBatchItems = useCurrentBatchItems();
  const items = useRecentItems();
  const { error, loadState } = useQueueLoadState();

  const excluded = useMemo(() => new Set(currentBatchItems.map((item) => item.id)), [currentBatchItems]);
  const filtered = useMemo(
    () => items.filter((item) => !excluded.has(item.id) && matchesFilter(item.status, filter)),
    [excluded, filter, items]
  );

  // Reveal requests we can't fulfill must not go stale: an item in the current
  // batch is already visible in NOW & NEXT, and one pruned past the recent
  // window is gone. Rows we do render consume the request themselves.
  useEffect(() => {
    if (revealItemId === null || loadState !== 'loaded') {
      return;
    }
    if (excluded.has(revealItemId) || !items.some((item) => item.id === revealItemId)) {
      clearPendingQueueItemReveal();
    }
  }, [excluded, items, loadState, revealItemId]);

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
            <QueueItemRow key={item.id} item={item} revealRequested={item.id === revealItemId} />
          ))}
        </Stack>
      )}
    </Stack>
  );
};
