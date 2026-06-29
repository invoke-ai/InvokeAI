import type { Project, WorkbenchPreferences } from '@workbench/types';

import { buildProjectQueueItemOriginPrefix, parseQueueItemOrigin } from '@workbench/backend/events';
import { useWorkbenchPreferences } from '@workbench/settings/store';
import { useOptionalWorkbenchSelector } from '@workbench/WorkbenchContext';
import { useMemo } from 'react';

import type { QueueQueryScope, QueueServerItem } from './queueServerApi';

import { useCurrentBatchItems, useQueueCounts, useRecentItems } from './queueDataStore';

export type QueueJobsScope = WorkbenchPreferences['queueJobsScope'];

export const isQueueServerItemInProject = (serverItem: QueueServerItem, project: Pick<Project, 'queue'>): boolean => {
  const localQueueItemId = parseQueueItemOrigin(serverItem.origin);

  if (localQueueItemId && project.queue.items.some((item) => item.id === localQueueItemId)) {
    return true;
  }

  return project.queue.items.some((item) => item.backendItemIds?.includes(serverItem.item_id) ?? false);
};

export const useQueueJobsScope = (): QueueJobsScope => useWorkbenchPreferences().queueJobsScope;

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
  const queueJobsScope = useQueueJobsScope();
  const projectId = useOptionalWorkbenchSelector((snapshot) => snapshot.activeProject.id, null);

  return useMemo(() => getQueueQueryScope({ projectId, queueJobsScope }), [projectId, queueJobsScope]);
};

export const useScopedQueueCounts = useQueueCounts;

export const useScopedCurrentBatchItems = useCurrentBatchItems;

export const useScopedRecentItems = useRecentItems;
