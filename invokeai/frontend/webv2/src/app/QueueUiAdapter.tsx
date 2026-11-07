import type { QueueItem } from '@features/queue/contracts';
import type { WorkbenchSnapshot } from '@workbench/workbenchStore';

import { useAuthSession, useCapabilities } from '@features/identity';
import { getModelLoads, subscribeModelLoads } from '@features/models';
import { getQueueItemSnapshotLabel } from '@features/queue/contracts';
import { getQueueItemAccess, QueueUiProvider, type QueueLocalRun, type QueueUiAdapter } from '@features/queue/react';
import { areArraysEqual, shallowEqual } from '@platform/state/selectors';
import { isUnsubmittedQueueItem } from '@workbench/canvasStagingView';
import { useWorkbenchPreferences } from '@workbench/settings/store';
import { useNotify } from '@workbench/useNotify';
import { useOpenWorkbenchWidget } from '@workbench/useOpenWorkbenchWidget';
import { useActiveProjectSelector, useWorkbenchCommands, useWorkbenchSelector } from '@workbench/WorkbenchContext';
import { lazy, useMemo, type ReactNode } from 'react';

const importQueueItemActions = () => import('@workbench/queue-integration/QueueItemActions');
const QueueItemActions = lazy(() => importQueueItemActions().then((module) => ({ default: module.QueueItemActions })));

const toQueueLocalRun = (projectId: string, item: QueueItem): QueueLocalRun => {
  const retry = item.submissionRetry;
  // Read defensively, as the snapshot readers are: a run restored from an older journal may lack parts of it.
  const submittedAt = Date.parse((item.snapshot as Partial<QueueItem['snapshot']> | undefined)?.submittedAt ?? '');

  return {
    id: item.id,
    label: getQueueItemSnapshotLabel(item),
    projectId,
    retry: retry ? { failureCount: retry.failureCount, message: retry.message, reason: retry.reason } : null,
    submittedAt: Number.isFinite(submittedAt) ? submittedAt : 0,
  };
};

const selectLocalRuns = (snapshot: WorkbenchSnapshot): QueueLocalRun[] =>
  snapshot.projects.flatMap((project) =>
    project.queue.items.filter(isUnsubmittedQueueItem).map((item) => toQueueLocalRun(project.id, item))
  );

/** Runs map to fresh objects on every Workbench change; equal fields keep the previous list and its consumers still. */
const areLocalRunsEqual = (left: QueueLocalRun[], right: QueueLocalRun[]): boolean =>
  left.length === right.length &&
  left.every((run, index) => {
    const { retry, ...fields } = run;
    const { retry: otherRetry, ...otherFields } = right[index]!;

    return shallowEqual(fields, otherFields) && shallowEqual(retry, otherRetry);
  });

/** Read by the queue's running row and status alone, so model loads re-render them rather than every queue consumer. */
const loadingModel = {
  getLabel: (): string | null => getModelLoads()[0]?.label ?? null,
  subscribe: subscribeModelLoads,
};

export const QueueUiAdapterProvider = ({ children }: { children: ReactNode }) => {
  const notify = useNotify();
  const { queue } = useWorkbenchCommands();
  const activeProjectId = useActiveProjectSelector((project) => project.id);
  const activeProjectName = useActiveProjectSelector((project) => project.name);
  const { canManageAppConfig, canManageModels } = useCapabilities();
  const session = useAuthSession();
  const { queueJobsScope } = useWorkbenchPreferences();
  const openWorkbenchWidget = useOpenWorkbenchWidget();
  const isConnected = useWorkbenchSelector((snapshot) => snapshot.backendConnection.status === 'connected');
  const openProjectIds = useWorkbenchSelector(
    (snapshot) => snapshot.projects.map((project) => project.id),
    areArraysEqual
  );
  const localRuns = useWorkbenchSelector(selectLocalRuns, areLocalRunsEqual);
  const adapter = useMemo<QueueUiAdapter>(() => {
    const viewer = {
      currentUserId: session.user?.user_id ?? null,
      isAdmin: session.user?.is_admin === true,
      multiuserEnabled: session.multiuserEnabled,
    };

    return {
      activeProjectId,
      activeProjectName,
      ItemActions: QueueItemActions,
      canManageInvocationCache: canManageAppConfig,
      canManageModelCache: canManageModels,
      canManageProcessor: canManageModels,
      canManageItem: (item) => getQueueItemAccess(item, viewer).canManage,
      canViewItemDetails: (item) => getQueueItemAccess(item, viewer).canViewDetails,
      cancelLocalRun: (projectId, runId) => queue.cancel(projectId, runId),
      cancelLocalRuns: (projectId) => queue.cancelUnsubmitted(projectId ?? undefined),
      isConnected,
      isProjectOpen: (projectId) => openProjectIds.includes(projectId),
      loadingModel,
      localRuns,
      notify,
      openQueue: () => openWorkbenchWidget('queue'),
      preloadItemActions: () => void importQueueItemActions(),
      queueJobsScope,
      viewer: !viewer.multiuserEnabled ? 'single' : viewer.isAdmin ? 'admin' : 'member',
    };
  }, [
    activeProjectId,
    activeProjectName,
    canManageAppConfig,
    canManageModels,
    isConnected,
    localRuns,
    notify,
    openProjectIds,
    openWorkbenchWidget,
    queue,
    queueJobsScope,
    session,
  ]);

  return <QueueUiProvider adapter={adapter}>{children}</QueueUiProvider>;
};
