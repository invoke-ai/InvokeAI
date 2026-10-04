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
