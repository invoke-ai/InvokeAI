import { useAuthSession, useCapabilities } from '@features/identity';
import { getQueueItemAccess, QueueUiProvider, type QueueUiAdapter } from '@features/queue/react';
import { areArraysEqual } from '@platform/state/selectors';
import { useWorkbenchPreferences } from '@workbench/settings/store';
import { useNotify } from '@workbench/useNotify';
import { useOpenWorkbenchWidget } from '@workbench/useOpenWorkbenchWidget';
import { useActiveProjectSelector, useWorkbenchSelector } from '@workbench/WorkbenchContext';
import { lazy, useMemo, type ReactNode } from 'react';

const importQueueItemActions = () => import('@workbench/queue-integration/QueueItemActions');
const QueueItemActions = lazy(() => importQueueItemActions().then((module) => ({ default: module.QueueItemActions })));

export const QueueUiAdapterProvider = ({ children }: { children: ReactNode }) => {
  const notify = useNotify();
  const activeProjectId = useActiveProjectSelector((project) => project.id);
  const { canManageAppConfig, canManageModels } = useCapabilities();
  const session = useAuthSession();
  const { queueJobsScope } = useWorkbenchPreferences();
  const openWorkbenchWidget = useOpenWorkbenchWidget();
  const isConnected = useWorkbenchSelector((snapshot) => snapshot.backendConnection.status === 'connected');
  const openProjectIds = useWorkbenchSelector(
    (snapshot) => snapshot.projects.map((project) => project.id),
    areArraysEqual
  );
  const adapter = useMemo<QueueUiAdapter>(() => {
    const viewer = {
      currentUserId: session.user?.user_id ?? null,
      isAdmin: session.user?.is_admin === true,
      multiuserEnabled: session.multiuserEnabled,
    };

    return {
      activeProjectId,
      ItemActions: QueueItemActions,
      canManageInvocationCache: canManageAppConfig,
      canManageProcessor: canManageModels,
      canManageItem: (item) => getQueueItemAccess(item, viewer).canManage,
      canViewItemDetails: (item) => getQueueItemAccess(item, viewer).canViewDetails,
      isConnected,
      isProjectOpen: (projectId) => openProjectIds.includes(projectId),
      notify,
      openQueue: () => openWorkbenchWidget('queue'),
      preloadItemActions: () => void importQueueItemActions(),
      queueJobsScope,
    };
  }, [
    activeProjectId,
    canManageAppConfig,
    canManageModels,
    isConnected,
    notify,
    openProjectIds,
    openWorkbenchWidget,
    queueJobsScope,
    session,
  ]);

  return <QueueUiProvider adapter={adapter}>{children}</QueueUiProvider>;
};
