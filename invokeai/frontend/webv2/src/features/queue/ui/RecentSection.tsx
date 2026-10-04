import { Stack, Text } from '@chakra-ui/react';
import { useMountEffect } from '@platform/react/useMountEffect';
import { segmentTabsPanelId, segmentTabsTabId } from '@platform/ui';
import { ListSectionHeader } from '@platform/ui/list/ListSectionHeader';
import { ListStack } from '@platform/ui/list/ListStack';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import type { QueueFilterId } from './queueFilters';

import { useCurrentBatchItems, useQueueLoadState, useRecentItems } from './queueDataStore';
import { matchesFilter } from './queueFilters';
import { QUEUE_FILTER_TABS_ID } from './QueueFilterTabs';
import { QueueItemRow } from './QueueItemRow';
import { clearPendingQueueItemReveal, type QueueItemRevealRequest } from './queueUiStore';

/** Exclude running and next items from recent history because NOW & NEXT already shows them. */
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
    !excluded.has(revealRequest.itemId) &&
    !items.some((item) => item.id === revealRequest.itemId);

  return (
    <Stack
      aria-labelledby={segmentTabsTabId(QUEUE_FILTER_TABS_ID, filter)}
      gap="1"
      id={segmentTabsPanelId(QUEUE_FILTER_TABS_ID)}
      // Row surfaces bleed into the widget padding so their content lines up with the tabs above.
      mx="-2"
      role="tabpanel"
    >
      {cannotReveal && revealRequest ? (
        <UnavailableRevealConsumer key={revealRequest.requestId} request={revealRequest} />
      ) : null}
      <ListSectionHeader count={filtered.length} label={t('common.recent')} />
      {filtered.length === 0 ? (
        <Text color={loadState === 'error' ? 'fg.error' : 'fg.subtle'} fontSize="xs" px="2">
          {loadState === 'loading'
            ? t('widgets.queue.loading')
            : loadState === 'error'
              ? error
              : t('common.nothingHereYet')}
        </Text>
      ) : (
        <ListStack dividers label={t('common.recent')}>
          {filtered.map((item) => (
            <QueueItemRow
              key={item.id}
              item={item}
              revealRequest={item.id === revealRequest?.itemId ? revealRequest : null}
            />
          ))}
        </ListStack>
      )}
    </Stack>
  );
};

const UnavailableRevealConsumer = ({ request }: { request: QueueItemRevealRequest }) => {
  useMountEffect(() => clearPendingQueueItemReveal(request.requestId));

  return null;
};
