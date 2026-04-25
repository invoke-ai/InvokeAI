import { Button, Flex } from '@invoke-ai/ui-library';
import { useStageFeedPagination } from 'features/gallery/hooks/useStageFeed';
import { memo } from 'react';

import type { StageViewProps } from './common';
import { StageViewerLinearBoardItem, StageViewerLinearQueueItem } from './StageViewerLinearItem'

/**
 * TODO:
 *
 * - Refactor this to make parameters the prominent list item, accompanied by images sharing them in their generation details
 */

export const StageViewerLinearView = memo(({ feed }: StageViewProps) => {
   const { visibleFeed, loadMore, hasMore, isLoading } = useStageFeedPagination(feed);

  return (
    <Flex flexDir="column">
      <Flex flexDir="column">
        {visibleFeed.map((item, index) => (
          item.type === 'board_item' ? <StageViewerLinearBoardItem key={index} item={item} /> : <StageViewerLinearQueueItem key={index} />
        ))}
      </Flex>
      {hasMore && (
        <Button isLoading={isLoading} onClick={loadMore}>Load More</Button>
      )}
    </Flex>
  );
});

StageViewerLinearView.displayName = 'StageViewerLinearView';
