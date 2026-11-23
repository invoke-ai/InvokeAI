import type { QueueItem } from '@features/queue/core/historyTypes';
import type { QueueItemReadModel, QueueQueryScope, QueueReadModel } from '@features/queue/core/types';
import type { TFunction } from 'i18next';

import { buildProjectQueueItemOriginPrefix, parseQueueItemOrigin } from '@features/queue/data/events';
import { useCallback, useMemo, useSyncExternalStore } from 'react';

import { useQueueUi, type QueueLocalRun, type QueueViewer } from './QueueUiContext';

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

/** Counted from the project's side so a run that just finished here, awaiting refetch, is not work elsewhere. */
export const getQueueOpenWorkElsewhere = (
  model: Pick<QueueReadModel, 'current' | 'items' | 'next'>,
  scopedOpen: number,
  project: { queue: { items: QueueItem[] } }
): number => {
  const seen = new Set<number>();
  let projectOpen = 0;

  for (const item of [model.current, model.next, ...model.items]) {
    if (!item || seen.has(item.id)) {
      continue;
    }

    seen.add(item.id);

    if ((item.status === 'pending' || item.status === 'in_progress') && isQueueItemReadModelInProject(item, project)) {
      projectOpen += 1;
    }
  }

  // A large run reaches past the newest-items window, so its own record of awaited backend items counts too.
  const projectOpenLocally = project.queue.items
    .filter((item) => item.status === 'pending' || item.status === 'running')
    .reduce((sum, item) => {
      const done = new Set([...(item.completedBackendItemIds ?? []), ...(item.cancelledBackendItemIds ?? [])]);

      return sum + (item.backendItemIds ?? []).filter((id) => !done.has(id)).length;
    }, 0);

  return Math.max(0, scopedOpen - Math.max(projectOpen, projectOpenLocally));
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

/** Same reach as the query scope and bulk actions; cancel-all counts all of them, shown or not. */
export const useScopedLocalRuns = (): readonly QueueLocalRun[] => {
  const { activeProjectId, localRuns, queueJobsScope } = useQueueUi();
  const projectId = queueJobsScope === 'active-project' ? activeProjectId : null;

  return useMemo(
    () => (projectId === null ? localRuns : localRuns.filter((run) => run.projectId === projectId)),
    [localRuns, projectId]
  );
};

/** Showing it sooner would flash a row on every Invoke before the backend's row replaces it. */
export const LOCAL_RUN_REVEAL_DELAY_MS = 1000;

const NO_LOCAL_RUNS: readonly QueueLocalRun[] = [];

/** When a run earns its row: at once after a failure or while offline, otherwise once its enqueue outlasts the delay. */
const getRevealAt = (run: QueueLocalRun, isConnected: boolean): number =>
  run.retry !== null || !isConnected ? Number.NEGATIVE_INFINITY : run.submittedAt + LOCAL_RUN_REVEAL_DELAY_MS;

/** A restored run is old, so it shows at once; one timer at a time re-renders as each comes due. */
export const useShownLocalRuns = (): readonly QueueLocalRun[] => {
  const { isConnected } = useQueueUi();
  const runs = useScopedLocalRuns();
  const revealTimes = useMemo(
    () => runs.map((run) => getRevealAt(run, isConnected)).sort((left, right) => left - right),
    [isConnected, runs]
  );
  // How many reveal times have passed: it changes only as one comes due, so the clock is read outside render.
  const getDueCount = useCallback(() => {
    const now = Date.now();
    const next = revealTimes.findIndex((time) => time > now);

    return next === -1 ? revealTimes.length : next;
  }, [revealTimes]);
  const subscribe = useCallback(
    (notify: () => void) => {
      let timer: ReturnType<typeof setTimeout> | undefined;
      const schedule = () => {
        const now = Date.now();
        const next = revealTimes.find((time) => time > now);

        if (next !== undefined) {
          // At most the delay: further off means the clock moved back, and past 2^31-1 ms setTimeout fires at once.
          timer = setTimeout(
            () => {
              notify();
              schedule();
            },
            Math.min(next - now, LOCAL_RUN_REVEAL_DELAY_MS)
          );
        }
      };

      schedule();
      return () => clearTimeout(timer);
    },
    [revealTimes]
  );
  const dueCount = useSyncExternalStore(subscribe, getDueCount, getDueCount);

  return useMemo(() => {
    if (dueCount === runs.length) {
      return runs;
    }

    if (dueCount === 0) {
      return NO_LOCAL_RUNS;
    }

    const latestDue = revealTimes[dueCount - 1]!;

    return runs.filter((run) => getRevealAt(run, isConnected) <= latestDue);
  }, [dueCount, isConnected, revealTimes, runs]);
};

/** A member's server counts nobody's history but the whole queue's, so their history counts are unknown. */
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

/** Translated whole as `<key>.<scope kind>`, so no translation stitches in a fragment; `count` null omits it. */
export const translateScoped = (t: TFunction, key: string, scope: QueueActionScope, count?: number | null): string =>
  t(`${key}.${scope.kind}`, {
    ...(count === null || count === undefined ? {} : { count }),
    ...(scope.kind === 'project' ? { project: scope.project } : {}),
  });
