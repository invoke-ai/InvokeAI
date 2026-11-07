import type { QueueSubmissionRetry } from '@features/queue/core/historyTypes';
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
  /** The invocation cache endpoints are admin-only; the Cache popover shows its row to administrators alone. */
  canManageInvocationCache: boolean;
  /** Model cache stats and clearing are admin-only too; without either cache the Cache button is absent. */
  canManageModelCache: boolean;
  canManageProcessor: boolean;
  canManageItem(item: QueueItemReadModel): boolean;
  canViewItemDetails(item: QueueItemReadModel): boolean;
  isConnected: boolean;
  /** Whether `projectId` is open in this tab; a Canvas rerun is staged only into a project open here. */
  isProjectOpen(projectId: string): boolean;
  /**
   * Runs this tab queued that the backend has not accepted yet (unsubmitted, enqueue in flight, or retried after
   * failures), across this tab's open projects. The backend has no row for them, so only this tab can show or cancel
   * them. Keep the array stable while its runs are unchanged: the queue widget and its actions re-render on a new one.
   */
  localRuns: readonly QueueLocalRun[];
  /** Cancels one of those runs. */
  cancelLocalRun(projectId: string, runId: string): void;
  /** Cancels those runs in one project or (null) every open project. */
  cancelLocalRuns(projectId: string | null): void;
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

/** A run this tab queued that the backend has not accepted yet, as the queue widget lists it. */
export interface QueueLocalRun {
  id: string;
  projectId: string;
  /** What it makes, in the user's words (workflow name or prompt); empty when it carries neither. */
  label: string;
  /** Epoch ms it was queued; 0 when unknown, which reads as long ago. */
  submittedAt: number;
  /** Why its latest enqueue failed while it is retried; null before any failure. */
  retry: Pick<QueueSubmissionRetry, 'failureCount' | 'message' | 'reason'> | null;
}

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
