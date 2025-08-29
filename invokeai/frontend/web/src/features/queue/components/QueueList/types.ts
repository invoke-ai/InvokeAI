import type { GetQueueItemIdsArgs } from 'services/api/types';

export type ListContext = {
  queryArgs: GetQueueItemIdsArgs;
  openQueueItems: number[];
  toggleQueueItem: (item_id: number) => void;
};
