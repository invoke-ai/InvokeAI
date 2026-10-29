import type { QueueItemReadModel } from '@features/queue/core/types';

import { createContext, useContext, useSyncExternalStore, type ComponentType, type ReactNode } from 'react';

export interface QueueUiNotificationPort {
  error(title: string, description?: string): void;
  info(title: string, description?: string): void;
  success(title: string, description?: string): void;
}

/** This UI port preserves dependency direction: Queue cannot import Workbench. */
export interface QueueUiAdapter {
  ItemActions: ComponentType<{ item: QueueItemReadModel }>;
  activeProjectId: string | null;
  /** Names the active-project scope in confirmations and results. */
  activeProjectName: string | null;
  /** The invocation cache endpoints are admin-only; the footer hides for everyone else. */
  canManageInvocationCache: boolean;
  canManageProcessor: boolean;
  canManageItem(item: QueueItemReadModel): boolean;
  canViewItemDetails(item: QueueItemReadModel): boolean;
  isConnected: boolean;
  /** Whether `projectId` is open in this tab; a Canvas rerun is staged only into a project open here. */
  isProjectOpen(projectId: string): boolean;
  /**
   * Runs this tab queued that the backend has not accepted yet (unsubmitted, or with their enqueue in flight), in the
   * active project and across this tab's open projects. A backend cancel cannot reach them.
   */
  localUnsubmittedRuns: { activeProject: number; allProjects: number };
  /** Cancels those runs, in one project or (null) every open project. */
  cancelLocalUnsubmittedRuns(projectId: string | null): void;
  notify: QueueUiNotificationPort;
  openQueue(): void;
  /**
   * Warms the `ItemActions` chunk ahead of the first expand. Optional: an
   * adapter that supplies `ItemActions` eagerly has nothing to preload.
   */
  preloadItemActions?(): void;
  queueJobsScope: 'active-project' | 'all';
  /**
   * The first model the server is loading now, or null: a running item with no progress yet waits for it. A store, so
   * only the components that read it (`useLoadingModelLabel`) re-render as loads come and go.
   */
  loadingModel: QueueLoadingModelPort;
  /**
   * Who bulk actions act for: single-user, an administrator (whose cancel-all and clear reach every user's items), or
   * a member (whose reach only their own).
   */
  viewer: QueueViewer;
}

export type QueueViewer = 'admin' | 'member' | 'single';

export interface QueueLoadingModelPort {
  getLabel(): string | null;
  subscribe(listener: () => void): () => void;
}

const QueueUiContext = createContext<QueueUiAdapter | null>(null);

export const QueueUiProvider = ({ adapter, children }: { adapter: QueueUiAdapter; children: ReactNode }) => (
  <QueueUiContext.Provider value={adapter}>{children}</QueueUiContext.Provider>
);

export const useLoadingModelLabel = (): string | null => {
  const { loadingModel } = useQueueUi();

  return useSyncExternalStore(loadingModel.subscribe, loadingModel.getLabel, loadingModel.getLabel);
};

export const useQueueUi = (): QueueUiAdapter => {
  const adapter = useContext(QueueUiContext);

  if (!adapter) {
    throw new Error('Queue UI requires an App-composed QueueUiProvider.');
  }

  return adapter;
};
