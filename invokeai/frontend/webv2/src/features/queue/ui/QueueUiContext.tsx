import type { QueueItemReadModel } from '@features/queue/core/types';

import { createContext, useContext, type ComponentType, type ReactNode } from 'react';

export interface QueueUiNotificationPort {
  error(title: string, description?: string): void;
  info(title: string, description?: string): void;
  success(title: string, description?: string): void;
}

/**
 * Queue's UI port. The context is a dependency-direction port (the feature
 * may not import workbench), not a test seam; no second adapter is expected.
 */
export interface QueueUiAdapter {
  ItemActions: ComponentType<{ item: QueueItemReadModel }>;
  activeProjectId: string | null;
  canManageProcessor: boolean;
  isConnected: boolean;
  notify: QueueUiNotificationPort;
  openQueue(): void;
  queueJobsScope: 'active-project' | 'all';
}

const QueueUiContext = createContext<QueueUiAdapter | null>(null);

export const QueueUiProvider = ({ adapter, children }: { adapter: QueueUiAdapter; children: ReactNode }) => (
  <QueueUiContext.Provider value={adapter}>{children}</QueueUiContext.Provider>
);

export const useQueueUi = (): QueueUiAdapter => {
  const adapter = useContext(QueueUiContext);

  if (!adapter) {
    throw new Error('Queue UI requires an App-composed QueueUiProvider.');
  }

  return adapter;
};
