import { useCapabilities } from '@features/identity';
import { QueueUiProvider, type QueueUiAdapter } from '@features/queue/react';
import { useWorkbenchPreferences } from '@workbench/settings/store';
import { useNotify } from '@workbench/useNotify';
import { useOpenWorkbenchWidget } from '@workbench/useOpenWorkbenchWidget';
import { useActiveProjectSelector, useWorkbenchSelector } from '@workbench/WorkbenchContext';
import { lazy, useMemo, type ReactNode } from 'react';

const QueueItemActions = lazy(() =>
  import('@workbench/queue-integration/QueueItemActions').then((module) => ({ default: module.QueueItemActions }))
);

/** App-owned adapter for Queue UI concerns that belong to Workbench. */
export const QueueUiAdapterProvider = ({ children }: { children: ReactNode }) => {
  const notify = useNotify();
  const activeProjectId = useActiveProjectSelector((project) => project.id);
  const { canManageModels } = useCapabilities();
  const { queueJobsScope } = useWorkbenchPreferences();
  const openWorkbenchWidget = useOpenWorkbenchWidget();
  const isConnected = useWorkbenchSelector((snapshot) => snapshot.backendConnection.status === 'connected');
  const adapter = useMemo<QueueUiAdapter>(
    () => ({
      activeProjectId,
      ItemActions: QueueItemActions,
      canManageProcessor: canManageModels,
      isConnected,
      notify,
      openQueue: () => openWorkbenchWidget('queue'),
      queueJobsScope,
    }),
    [activeProjectId, canManageModels, isConnected, notify, openWorkbenchWidget, queueJobsScope]
  );

  return <QueueUiProvider adapter={adapter}>{children}</QueueUiProvider>;
};
