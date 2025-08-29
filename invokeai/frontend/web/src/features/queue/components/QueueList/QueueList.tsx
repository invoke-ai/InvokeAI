import { Flex, Heading, ListItem } from '@invoke-ai/ui-library';
import { IAINoContentFallbackWithSpinner } from 'common/components/IAIImageFallback';
import { useRangeBasedQueueItemFetching } from 'features/queue/hooks/useRangeBasedQueueItemFetching';
import { memo, useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import type {
  Components,
  Components,
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

const QueueItemAtPosition = memo(
  ({ index, itemId, context }: { index: number; itemId: number; context: ListContext }) => {
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

    return <QueueItemComponent index={index} item={queueItem} context={context} />;
  }
);
QueueItemAtPosition.displayName = 'QueueItemAtPosition';

const computeItemKey: ComputeItemKey<number, ListContext> = (index, itemId, { queryArgs }) => {
  return `${JSON.stringify(queryArgs)}-${itemId ?? index}`;
};

const itemContent: ItemContent<number, ListContext> = (index, itemId, context) => (
  <QueueItemAtPosition index={index} itemId={itemId} context={context} />
);

const ScrollSeekPlaceholderComponent: Components<ListContext>['ScrollSeekPlaceholder'] = (props) => (
  <ListItem aspectRatio="1/1" {...props}>
    <QueueItemPlaceholder />
  </ListItem>
);

ScrollSeekPlaceholderComponent.displayName = 'ScrollSeekPlaceholderComponent';

const components: Components<number, ListContext> = {
  List: QueueListComponent,
  // ScrollSeekPlaceholder: ScrollSeekPlaceholderComponent,
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

  const [openQueueItems, setOpenQueueItems] = useState<number[]>([]);

  const toggleQueueItem = useCallback((item_id: number) => {
    setOpenQueueItems((prev) => {
      if (prev.includes(item_id)) {
        return prev.filter((id) => id !== item_id);
      }
      return [...prev, item_id];
    });
  }, []);

  const context = useMemo<ListContext>(
    () => ({ queryArgs, openQueueItems, toggleQueueItem }),
    [queryArgs, openQueueItems, toggleQueueItem]
  );

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
        <Virtuoso<number, ListContext>
          ref={virtuosoRef}
          context={context}
          data={itemIds}
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
