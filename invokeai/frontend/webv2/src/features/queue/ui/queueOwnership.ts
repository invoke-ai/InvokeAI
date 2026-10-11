import type { QueueItemReadModel } from '@features/queue/core/types';

export interface QueueViewer {
  currentUserId: string | null;
  isAdmin: boolean;
  multiuserEnabled: boolean;
}

export interface QueueItemAccess {
  canManage: boolean;
  canViewDetails: boolean;
}

/** Interprets owner metadata and the backend's explicit redaction marker. */
export const getQueueItemAccess = (item: QueueItemReadModel, viewer: QueueViewer): QueueItemAccess => ({
  canManage:
    !viewer.multiuserEnabled ||
    viewer.isAdmin ||
    (viewer.currentUserId !== null && item.userId === viewer.currentUserId),
  canViewDetails: item.userId !== 'redacted',
});

/** Only failed or canceled roots retry: the backend clones the root session, so a child offers nothing extra. */
export const isRetryableQueueItem = (item: Pick<QueueItemReadModel, 'parentItemId' | 'status'>): boolean =>
  (item.status === 'failed' || item.status === 'canceled') &&
  (item.parentItemId === null || item.parentItemId === undefined);

/** Pending, waiting (a workflow-call parent) and running items can all be canceled. */
export const isCancellableQueueItem = (item: Pick<QueueItemReadModel, 'status'>): boolean =>
  item.status === 'pending' || item.status === 'waiting' || item.status === 'in_progress';
