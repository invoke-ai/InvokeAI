import { z } from 'zod';

export const zStageViewerMode = z.enum(['grid', 'linear']);

export const STAGE_FEED_QUEUE_ITEM_STATUS = ['pending', 'in_progress', 'failed'] as const;

export type StageFeedQueueItemStatus = typeof STAGE_FEED_QUEUE_ITEM_STATUS[number];

type StageFeedQueueItem = {
  type: 'queue_item';
  id: number;
  status: StageFeedQueueItemStatus;
  createdAt: string;
}

type StageFeedBoardItem = {
  type: 'board_item';
  id: string; // id is the image name
}

export type StageFeedItem =
  | StageFeedQueueItem
  | StageFeedBoardItem

export type StageFeed = StageFeedItem[];

export type StageViewProps = {
  feed: StageFeed;
};
