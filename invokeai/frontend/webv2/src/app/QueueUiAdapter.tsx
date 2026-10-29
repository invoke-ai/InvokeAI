import { useAuthSession, useCapabilities } from '@features/identity';
import { getModelLoads, subscribeModelLoads } from '@features/models';
import { getQueueItemAccess, QueueUiProvider, type QueueUiAdapter } from '@features/queue/react';
import { areArraysEqual } from '@platform/state/selectors';
import { isUnsubmittedQueueItem } from '@workbench/canvasStagingView';
import { useWorkbenchPreferences } from '@workbench/settings/store';
import { useNotify } from '@workbench/useNotify';
import { useOpenWorkbenchWidget } from '@workbench/useOpenWorkbenchWidget';
import { useActiveProjectSelector, useWorkbenchCommands, useWorkbenchSelector } from '@workbench/WorkbenchContext';
import { lazy, useMemo, type ReactNode } from 'react';

const importQueueItemActions = () => import('@workbench/queue-integration/QueueItemActions');
const QueueItemActions = lazy(() => importQueueItemActions().then((module) => ({ default: module.QueueItemActions })));

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
  const localUnsubmittedRuns = useWorkbenchSelector((snapshot) => {
    const count = (items: readonly Parameters<typeof isUnsubmittedQueueItem>[0][]) =>
      items.filter(isUnsubmittedQueueItem).length;

    return {
      activeProject: count(snapshot.activeProject.queue.items),
      allProjects: snapshot.projects.reduce((total, project) => total + count(project.queue.items), 0),
    };
  });
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
      canManageProcessor: canManageModels,
      canManageItem: (item) => getQueueItemAccess(item, viewer).canManage,
      canViewItemDetails: (item) => getQueueItemAccess(item, viewer).canViewDetails,
      cancelLocalUnsubmittedRuns: (projectId) => queue.cancelUnsubmitted(projectId ?? undefined),
      isConnected,
      isProjectOpen: (projectId) => openProjectIds.includes(projectId),
      loadingModel,
      localUnsubmittedRuns,
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
    localUnsubmittedRuns,
    notify,
    openProjectIds,
    openWorkbenchWidget,
    queue,
    queueJobsScope,
    session,
  ]);

  return <QueueUiProvider adapter={adapter}>{children}</QueueUiProvider>;
};
