import { STAGE_FEED_QUEUE_ITEM_STATUS, type StageFeed, type StageFeedQueueItemStatus } from 'features/gallery/components/StageViewer/common';
import { useGalleryImageNames } from 'features/gallery/components/use-gallery-image-names';
import { useCallback, useMemo, useState } from 'react';
import { queueApi } from 'services/api/endpoints/queue';

const PAGE_SIZE = 50;

export const useStageFeed = () => {
  const { imageNames, isLoading: isGalleryLoading } = useGalleryImageNames();
  const { data: queueItems } = queueApi.endpoints.listAllQueueItems.useQuery({
    destination: 'generate',
  })

  const feed = useMemo(() => {
    const activeQueueItems = (queueItems ?? []).filter((item) => STAGE_FEED_QUEUE_ITEM_STATUS.includes(item.status as StageFeedQueueItemStatus));

    const mappedQueueItems = activeQueueItems.map((item) => ({
      type: 'queue_item' as const,
      id: item.item_id,
      status: item.status as StageFeedQueueItemStatus,
      createdAt: item.created_at,
    }));


    const mappedBoardItems = (imageNames ?? []).map((name) => ({
      type: 'board_item' as const,
      id: name,
    }));

    return [
      ...mappedQueueItems,
      ...mappedBoardItems,
    ]
  }, [imageNames, queueItems]);

  return {
    feed,
    isLoading: isGalleryLoading,
  };
}

/**
 * TODO:
 *
 * - Tie the page to global state so that it persists across view mode changes and unmounts
 */

export const useStageFeedPagination = (fullFeed: StageFeed) => {
  const [page, setPage] = useState(1);
  const [isLoading, setIsLoading] = useState(false);

  const visibleFeed = useMemo(() => {
    return fullFeed.slice(0, page * PAGE_SIZE);
  }, [fullFeed, page]);

  const hasMore = useMemo(() => {
    return visibleFeed.length < fullFeed.length;
  }, [visibleFeed, fullFeed]);

  const loadMore = useCallback(() => {
    if (hasMore) {
      setIsLoading(true);
      setPage((p) => p + 1);
    }
    setIsLoading(false);
  }, [hasMore])

  return {
    hasMore,
    loadMore,
    isLoading,
    visibleFeed,
  }
}
