import { Flex, Heading } from '@invoke-ai/ui-library';
import { IAINoContentFallbackWithSpinner } from 'common/components/IAIImageFallback';
import { useRangeBasedQueueItemFetching } from 'features/queue/hooks/useRangeBasedQueueItemFetching';
import { memo, useCallback, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import type {
  Components,
  ComputeItemKey,
  ItemContent,
  ListRange,
  ScrollSeekConfiguration,
  VirtuosoHandle,
} from 'react-virtuoso';
import { Virtuoso } from 'react-virtuoso';
import { queueApi } from 'services/api/endpoints/queue';

import QueueItemComponent, { QueueItemPlaceholder } from './QueueItemComponent';
import QueueListComponent from './QueueListComponent';
import QueueListHeader from './QueueListHeader';
import type { ListContext } from './types';
import { useQueueItemIds } from './useQueueItemIds';
import { useScrollableQueueList } from './useScrollableQueueList';

const QueueItemAtPosition = memo(({ index, itemId }: { index: number; itemId: number }) => {
  /*
   * We rely on the useRangeBasedQueueItemFetching to fetch all queue items, caching them with RTK Query.
   *
   * In this component, we just want to consume that cache. Unforutnately, RTK Query does not provide a way to
   * subscribe to a query without triggering a new fetch.
   *
   * There is a hack, though:
   * - https://github.com/reduxjs/redux-toolkit/discussions/4213
   *
   * This essentially means "subscribe to the query once it has some data".
   */

  // Use `currentData` instead of `data` to prevent a flash of previous queue item rendered at this index
  const { currentData: queueItem, isUninitialized } = queueApi.endpoints.getQueueItem.useQueryState(itemId);
  queueApi.endpoints.getQueueItem.useQuerySubscription(itemId, { skip: isUninitialized });

  if (!queueItem) {
    return <QueueItemPlaceholder item-id={itemId} />;
  }

  return <QueueItemComponent index={index} item={queueItem} />;
});
QueueItemAtPosition.displayName = 'QueueItemAtPosition';

const computeItemKey: ComputeItemKey<number, ListContext> = (index: number, itemId: number, context: ListContext) => {
  return `${JSON.stringify(context.queryArgs)}-${itemId ?? index}`;
};

const itemContent: ItemContent<number, ListContext> = (index, itemId) => (
  <QueueItemAtPosition index={index} itemId={itemId} />
);

const ScrollSeekPlaceholderComponent: Components<number, ListContext>['ScrollSeekPlaceholder'] = (_props) => {
  return (
    <Flex>
      <QueueItemPlaceholder />
    </Flex>
  );
};

ScrollSeekPlaceholderComponent.displayName = 'ScrollSeekPlaceholderComponent';

const components: Components<number, ListContext> = {
  List: QueueListComponent,
  ScrollSeekPlaceholder: ScrollSeekPlaceholderComponent,
};

const scrollSeekConfiguration: ScrollSeekConfiguration = {
  enter: (velocity) => {
    return Math.abs(velocity) > 2048;
  },
  exit: (velocity) => {
    return velocity === 0;
  },
};

export const QueueList = () => {
  const virtuosoRef = useRef<VirtuosoHandle>(null);
  const rangeRef = useRef<ListRange>({ startIndex: 0, endIndex: 0 });
  const rootRef = useRef<HTMLDivElement>(null);
  const { t } = useTranslation();

  // Get the ordered list of queue item ids - this is our primary data source for virtualization
  const { queryArgs, itemIds, isLoading } = useQueueItemIds();

  // Use range-based fetching for bulk loading queue items into cache based on the visible range
  const { onRangeChanged } = useRangeBasedQueueItemFetching({
    itemIds,
    enabled: !isLoading,
  });

  const scrollerRef = useScrollableQueueList(rootRef) as (ref: HTMLElement | Window | null) => void;

  /*
   * We have to keep track of the visible range for keep-selected-image-in-view functionality and push the range to
   * the range-based queue item fetching hook.
   */
  const handleRangeChanged = useCallback(
    (range: ListRange) => {
      rangeRef.current = range;
      onRangeChanged(range);
    },
    [onRangeChanged]
  );

  const context = useMemo<ListContext>(() => ({ queryArgs }), [queryArgs]);

  if (isLoading) {
    return <IAINoContentFallbackWithSpinner />;
  }

  if (!itemIds.length) {
    return (
      <Flex w="full" h="full" alignItems="center" justifyContent="center">
        <Heading color="base.500">{t('queue.queueEmpty')}</Heading>
      </Flex>
    );
  }

  return (
    <Flex w="full" h="full" flexDir="column">
      <QueueListHeader />
      <Flex ref={rootRef} w="full" h="full" alignItems="center" justifyContent="center">
        <Virtuoso<number>
          ref={virtuosoRef}
          context={context}
          data={itemIds}
          increaseViewportBy={512}
          itemContent={itemContent}
          computeItemKey={computeItemKey}
          components={components}
          scrollerRef={scrollerRef}
          scrollSeekConfiguration={scrollSeekConfiguration}
          rangeChanged={handleRangeChanged}
        />
      </Flex>
    </Flex>
  );
};
