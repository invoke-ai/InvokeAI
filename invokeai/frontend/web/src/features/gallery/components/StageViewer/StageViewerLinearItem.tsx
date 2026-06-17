import { Flex } from "@invoke-ai/ui-library";
import { memo } from "react";

import type { StageFeedBoardItem } from './common';

export const StageViewerLinearBoardItem = memo(({ item }: { item: StageFeedBoardItem }) => {
  return (
    <Flex>
      Board Item: {item.id}
    </Flex>
  );
});

StageViewerLinearBoardItem.displayName = 'StageViewerLinearBoardItem';

// TODO

export const StageViewerLinearQueueItem = memo(() => {
  return (
    <Flex>
      Loading...
    </Flex>
  )
});

StageViewerLinearQueueItem.displayName = 'StageViewerLinearQueueItem';
