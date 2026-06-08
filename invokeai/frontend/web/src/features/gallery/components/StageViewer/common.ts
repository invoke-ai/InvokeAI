import { z } from 'zod';

export const zStageViewerMode = z.enum(['grid', 'linear']);

export const STAGE_FEED_QUEUE_ITEM_STATUS = ['pending', 'in_progress', 'failed'] as const;

export type StageFeedQueueItemStatus = typeof STAGE_FEED_QUEUE_ITEM_STATUS[number];

export type StageFeedQueueItem = {
  type: 'queue_item';
  /** queue id */
  id: number;
  status: StageFeedQueueItemStatus;
  createdAt: string;
}

export type StageFeedBoardItem = {
  type: 'board_item';
  /** image_name */
  id: string;
}

export type StageFeedItem =
  | StageFeedQueueItem
  | StageFeedBoardItem

export type StageFeed = StageFeedItem[];

export type StageViewProps = {
  feed: StageFeed;
};
