import { useAuthSession, useCapabilities } from '@features/identity';
import { getQueueItemAccess, QueueUiProvider, type QueueUiAdapter } from '@features/queue/react';
import { useWorkbenchPreferences } from '@workbench/settings/store';
import { useNotify } from '@workbench/useNotify';
import { useOpenWorkbenchWidget } from '@workbench/useOpenWorkbenchWidget';
import { useActiveProjectSelector, useWorkbenchSelector } from '@workbench/WorkbenchContext';
import { lazy, useMemo, type ReactNode } from 'react';

const QueueItemActions = lazy(() =>
  import('@workbench/queue-integration/QueueItemActions').then((module) => ({ default: module.QueueItemActions }))
);

/**
 * Production binding of Queue's UI port: adapts the Workbench-owned concerns
 * Queue's UI depends on. No second adapter is expected.
 */
export const QueueUiAdapterProvider = ({ children }: { children: ReactNode }) => {
  const notify = useNotify();
  const activeProjectId = useActiveProjectSelector((project) => project.id);
  const { canManageModels } = useCapabilities();
  const session = useAuthSession();
  const { queueJobsScope } = useWorkbenchPreferences();
  const openWorkbenchWidget = useOpenWorkbenchWidget();
  const isConnected = useWorkbenchSelector((snapshot) => snapshot.backendConnection.status === 'connected');
  const adapter = useMemo<QueueUiAdapter>(() => {
    const viewer = {
      currentUserId: session.user?.user_id ?? null,
      isAdmin: session.user?.is_admin === true,
      multiuserEnabled: session.multiuserEnabled,
    };

    return {
      activeProjectId,
      ItemActions: QueueItemActions,
      canManageProcessor: canManageModels,
      canManageItem: (item) => getQueueItemAccess(item, viewer).canManage,
      canViewItemDetails: (item) => getQueueItemAccess(item, viewer).canViewDetails,
      isConnected,
      notify,
      openQueue: () => openWorkbenchWidget('queue'),
      queueJobsScope,
    };
  }, [activeProjectId, canManageModels, isConnected, notify, openWorkbenchWidget, queueJobsScope, session]);

  return <QueueUiProvider adapter={adapter}>{children}</QueueUiProvider>;
};
