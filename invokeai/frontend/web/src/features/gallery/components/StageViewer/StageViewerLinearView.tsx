import { Button, Flex, Text } from '@invoke-ai/ui-library';
import { useStageFeedPagination } from 'features/gallery/hooks/useStageFeed';
import { memo } from 'react';

import type { StageViewProps } from './common';

export const StageViewerLinearView = memo(({ feed }: StageViewProps) => {
   const { visibleFeed, loadMore, hasMore, isLoading } = useStageFeedPagination(feed);

  return (
    <Flex flexDir="column">
      <Flex flexDir="column">
        {visibleFeed.map((item, index) => (
          <Flex
            key={index}
          >
            <Text>{item.type}</Text>
          </Flex>
        ))}
      </Flex>
      {hasMore && (
        <Button isLoading={isLoading} onClick={loadMore}>Load More</Button>
      )}
    </Flex>
  );
});

StageViewerLinearView.displayName = 'StageViewerLinearView';
