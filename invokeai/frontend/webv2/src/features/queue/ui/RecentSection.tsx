import { Stack, Text } from '@chakra-ui/react';
import { useMountEffect } from '@platform/react/useMountEffect';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import type { QueueFilterId } from './queueFilters';

import { useCurrentBatchItems, useQueueLoadState, useRecentItems } from './queueDataStore';
import { matchesFilter } from './queueFilters';
import { QueueItemRow } from './QueueItemRow';
import { clearPendingQueueItemReveal, type QueueItemRevealRequest } from './queueUiStore';
import { SectionHeader } from './SectionHeader';

/**
 * RECENT — the windowed queue history, filtered by the active tab. The running
 * and next items are excluded here since NOW & NEXT already shows them.
 */
export const RecentSection = ({
  filter,
  revealRequest = null,
}: {
  filter: QueueFilterId;
  revealRequest?: QueueItemRevealRequest | null;
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

  const cannotReveal =
    revealRequest !== null &&
    loadState === 'loaded' &&
    (excluded.has(revealRequest.itemId) || !items.some((item) => item.id === revealRequest.itemId));

  return (
    <Stack gap="2">
      {cannotReveal && revealRequest ? (
        <UnavailableRevealConsumer key={revealRequest.requestId} request={revealRequest} />
      ) : null}
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
            <QueueItemRow
              key={item.id}
              item={item}
              revealRequest={item.id === revealRequest?.itemId ? revealRequest : null}
            />
          ))}
        </Stack>
      )}
    </Stack>
  );
};

const UnavailableRevealConsumer = ({ request }: { request: QueueItemRevealRequest }) => {
  useMountEffect(() => clearPendingQueueItemReveal(request.requestId));

  return null;
};
