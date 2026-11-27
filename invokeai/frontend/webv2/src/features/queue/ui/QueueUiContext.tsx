import type { QueueSubmissionRetry } from '@features/queue/core/historyTypes';
import type { QueueItemReadModel } from '@features/queue/core/types';
import type { MenuPartLoader } from '@platform/ui/Menu';

import { createContext, useContext, useSyncExternalStore, type ComponentType, type ReactNode } from 'react';

export interface QueueUiNotificationPort {
  error(title: string, description?: string): void;
  info(title: string, description?: string): void;
  success(title: string, description?: string): void;
}

/** Queue cannot import Workbench. */
export interface QueueUiAdapter {
  ItemActions: ComponentType<{ item: QueueItemReadModel }>;
  /** Mounted only while that row's menu is open. */
  itemRecallMenuGroup: MenuPartLoader<ComponentType<{ item: QueueItemReadModel }>>;
  activeProjectId: string | null;
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
  /** The backend has no row for these, so only this tab can show or cancel them; keep the array stable. */
  localRuns: readonly QueueLocalRun[];
  cancelLocalRun(projectId: string, runId: string): void;
  /** Cancels those runs in one project or (null) every open project. */
  cancelLocalRuns(projectId: string | null): void;
  notify: QueueUiNotificationPort;
  openQueue(): void;
  /** Optional: an adapter that supplies `ItemActions` eagerly has nothing to preload. */
  preloadItemActions?(): void;
  queueJobsScope: 'active-project' | 'all';
  recallAllFromItem(item: QueueItemReadModel): void;
  /** A store, so only components that read it re-render as loads come and go. */
  loadingModel: QueueLoadingModelPort;
  /** An administrator's cancel-all and clear reach every user's items; a member's only their own. */
  viewer: QueueViewer;
}

export type QueueViewer = 'admin' | 'member' | 'single';

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
