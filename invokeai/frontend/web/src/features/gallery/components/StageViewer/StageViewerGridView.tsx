import { Box, Button, Flex } from '@invoke-ai/ui-library';
import { useStageFeedPagination } from 'features/gallery/hooks/useStageFeed';
import { memo } from 'react';

import type { StageViewProps } from './common';
import { StageViewerGridBoardItem, StageViewerGridQueueItem } from './StageViewerGridItem'

/**
 * TODO:
 *
 * - Use Colcade for masonry layout
 */

export const StageViewerGridView = memo(({ feed }: StageViewProps) => {
   const { visibleFeed, loadMore, hasMore, isLoading } = useStageFeedPagination(feed);

  return (
    <Flex flexDir="column" w="full" h="full" overflowY="auto">
      <Box w="full" display="grid" gridTemplateColumns="repeat(auto-fill, minmax(200px, 1fr))" gap={4} p={2}>
        {visibleFeed.map((item, index) => (
          item.type === 'board_item' ? <StageViewerGridBoardItem key={index} item={item} /> : <StageViewerGridQueueItem key={index} item={item} />
        ))}
      </Box>

      {/* Pagination */}
      {hasMore && (
        <Flex w="full" justifyContent="center" p={4}>
          <Button isLoading={isLoading} onClick={loadMore}>Load More</Button>
        </Flex>
      )}
    </Flex>
  );
});

StageViewerGridView.displayName = 'StageViewerGridView';
