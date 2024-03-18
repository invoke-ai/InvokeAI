import { Flex, Heading } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallbackWithSpinner } from 'common/components/IAIImageFallback';
import { overlayScrollbarsParams } from 'common/components/OverlayScrollbars/constants';
import { listCursorChanged, listPriorityChanged } from 'features/queue/store/queueSlice';
import { useOverlayScrollbars } from 'overlayscrollbars-react';
import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import type { Components, ItemContent } from 'react-virtuoso';
import { Virtuoso } from 'react-virtuoso';
import { queueItemsAdapterSelectors, useListQueueItemsQuery } from 'services/api/endpoints/queue';
import type { SessionQueueItemDTO } from 'services/api/types';

import QueueItemComponent from './QueueItemComponent';
import QueueListComponent from './QueueListComponent';
import QueueListHeader from './QueueListHeader';
import type { ListContext } from './types';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type TableVirtuosoScrollerRef = (ref: HTMLElement | Window | null) => any;

const computeItemKey = (index: number, item: SessionQueueItemDTO): number => item.item_id;

const components: Components<SessionQueueItemDTO, ListContext> = {
  List: QueueListComponent,
};

const itemContent: ItemContent<SessionQueueItemDTO, ListContext> = (index, item, context) => (
  <QueueItemComponent index={index} item={item} context={context} />
);

const QueueList = () => {
  const listCursor = useAppSelector((s) => s.queue.listCursor);
  const listPriority = useAppSelector((s) => s.queue.listPriority);
  const dispatch = useAppDispatch();
  const rootRef = useRef<HTMLDivElement>(null);
  const [scroller, setScroller] = useState<HTMLElement | null>(null);
  const [initialize, osInstance] = useOverlayScrollbars(overlayScrollbarsParams);
  const { t } = useTranslation();

  useEffect(() => {
    const { current: root } = rootRef;
    if (scroller && root) {
      initialize({
        target: root,
        elements: {
          viewport: scroller,
        },
      });
    }
    return () => osInstance()?.destroy();
  }, [scroller, initialize, osInstance]);

  const { data: listQueueItemsData, isLoading } = useListQueueItemsQuery({
    cursor: listCursor,
    priority: listPriority,
  });

  const queueItems = useMemo(() => {
    if (!listQueueItemsData) {
      return [];
    }
    return queueItemsAdapterSelectors.selectAll(listQueueItemsData);
  }, [listQueueItemsData]);

  const handleLoadMore = useCallback(() => {
    if (!listQueueItemsData?.has_more) {
      return;
    }
    const lastItem = queueItems[queueItems.length - 1];
    if (!lastItem) {
      return;
    }
    dispatch(listCursorChanged(lastItem.item_id));
    dispatch(listPriorityChanged(lastItem.priority));
  }, [dispatch, listQueueItemsData?.has_more, queueItems]);

  const [openQueueItems, setOpenQueueItems] = useState<number[]>([]);

  const toggleQueueItem = useCallback((item_id: number) => {
    setOpenQueueItems((prev) => {
      if (prev.includes(item_id)) {
        return prev.filter((id) => id !== item_id);
      }
      return [...prev, item_id];
    });
  }, []);

  const context = useMemo<ListContext>(() => ({ openQueueItems, toggleQueueItem }), [openQueueItems, toggleQueueItem]);

  if (isLoading) {
    return <IAINoContentFallbackWithSpinner />;
  }

  if (!queueItems.length) {
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
        <Virtuoso<SessionQueueItemDTO, ListContext>
          data={queueItems}
          endReached={handleLoadMore}
          scrollerRef={setScroller as TableVirtuosoScrollerRef}
          itemContent={itemContent}
          computeItemKey={computeItemKey}
          components={components}
          context={context}
        />
      </Flex>
    </Flex>
  );
};

export default memo(QueueList);
