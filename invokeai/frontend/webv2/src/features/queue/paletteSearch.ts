import type { QueueQueryScope, QueueReadModel } from './core/types';

import { queueBackend } from './data/httpRealtimeQueueBackend';

/** Reads the bounded queue model used by deferred command-palette search. */
export const loadPaletteQueueReadModel = async (scope: QueueQueryScope): Promise<QueueReadModel> => {
  const [status, current, next, idsResult] = await Promise.all([
    queueBackend.readStatus(scope),
    queueBackend.readCurrent(scope),
    queueBackend.readNext(scope),
    queueBackend.readItemIds('desc', scope),
  ]);
  const items = await queueBackend.readItemsById(idsResult.itemIds.slice(0, 50));

  return { current, items, next, scope, status };
};
