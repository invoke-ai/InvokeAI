import type { QueueItem } from '@features/queue/core/historyTypes';
import type { QueueItemReadModel, QueueQueryScope } from '@features/queue/core/types';

import { buildProjectQueueItemOriginPrefix, parseQueueItemOrigin } from '@features/queue/data/events';
import { useMemo } from 'react';

import { useQueueUi } from './QueueUiContext';

export type QueueJobsScope = 'active-project' | 'all';

export const isQueueItemReadModelInProject = (
  serverItem: QueueItemReadModel,
  project: { queue: { items: QueueItem[] } }
): boolean => {
  const localQueueItemId = parseQueueItemOrigin(serverItem.origin);

  if (localQueueItemId && project.queue.items.some((item) => item.id === localQueueItemId)) {
    return true;
  }

  return project.queue.items.some((item) => item.backendItemIds?.includes(serverItem.id) ?? false);
};

export const getQueueQueryScope = ({
  projectId,
  queueJobsScope,
}: {
  projectId: string | null;
  queueJobsScope: QueueJobsScope;
}): QueueQueryScope =>
  queueJobsScope === 'active-project' && projectId
    ? { originPrefix: buildProjectQueueItemOriginPrefix(projectId) }
    : {};

export const useQueueQueryScope = (): QueueQueryScope => {
  const { activeProjectId: projectId, queueJobsScope } = useQueueUi();

  return useMemo(() => getQueueQueryScope({ projectId, queueJobsScope }), [projectId, queueJobsScope]);
};
