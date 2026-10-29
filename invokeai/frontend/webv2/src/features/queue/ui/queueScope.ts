import type { QueueItem } from '@features/queue/core/historyTypes';
import type { QueueItemReadModel, QueueQueryScope } from '@features/queue/core/types';
import type { TFunction } from 'i18next';

import { buildProjectQueueItemOriginPrefix, parseQueueItemOrigin } from '@features/queue/data/events';
import { useMemo } from 'react';

import { useQueueUi, type QueueViewer } from './QueueUiContext';

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

/**
 * The reach of a bulk action as the viewer meets it: one project; every project (single-user); every project and user
 * (an administrator, whose cancel and clear reach everyone's items); or every project of the viewer's own (a member,
 * for whom the server counts nobody's history but the whole queue's, so history counts are unknown).
 */
export type QueueActionScope = { kind: 'all' | 'allUsers' | 'mine' } | { kind: 'project'; project: string };

export const getQueueActionScope = ({
  activeProjectId,
  activeProjectName,
  queueJobsScope,
  viewer,
}: {
  activeProjectId: string | null;
  activeProjectName: string | null;
  queueJobsScope: QueueJobsScope;
  viewer: QueueViewer;
}): QueueActionScope => {
  if (queueJobsScope === 'active-project' && activeProjectId) {
    return { kind: 'project', project: activeProjectName ?? '' };
  }

  return { kind: viewer === 'admin' ? 'allUsers' : viewer === 'member' ? 'mine' : 'all' };
};

export const useQueueActionScope = (): QueueActionScope => {
  const { activeProjectId, activeProjectName, queueJobsScope, viewer } = useQueueUi();

  return useMemo(
    () => getQueueActionScope({ activeProjectId, activeProjectName, queueJobsScope, viewer }),
    [activeProjectId, activeProjectName, queueJobsScope, viewer]
  );
};

/**
 * A sentence that names the scope, translated whole: `<key>.<scope kind>`, so no translation stitches a fragment in.
 * `count` null leaves the count out, for the variants that cannot know it.
 */
export const translateScoped = (t: TFunction, key: string, scope: QueueActionScope, count?: number | null): string =>
  t(`${key}.${scope.kind}`, {
    ...(count === null || count === undefined ? {} : { count }),
    ...(scope.kind === 'project' ? { project: scope.project } : {}),
  });
