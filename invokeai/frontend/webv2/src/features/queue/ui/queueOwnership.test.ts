import type { QueueItemReadModel } from '@features/queue/core/types';

import { buildUtilityQueueItemOrigin } from '@features/queue/data/events';
import { describe, expect, it } from 'vitest';

import { getQueueItemAccess, isCancellableQueueItem, isRetryableQueueItem } from './queueOwnership';

const item = { userId: 'owner' } as QueueItemReadModel;

describe('queue item ownership', () => {
  it('lets owners and admins manage an item', () => {
    expect(getQueueItemAccess(item, { currentUserId: 'owner', isAdmin: false, multiuserEnabled: true }).canManage).toBe(
      true
    );
    expect(getQueueItemAccess(item, { currentUserId: 'admin', isAdmin: true, multiuserEnabled: true }).canManage).toBe(
      true
    );
  });

  it('blocks foreign users from management and redacted details', () => {
    expect(
      getQueueItemAccess(
        { ...item, userId: 'redacted' },
        { currentUserId: 'viewer', isAdmin: false, multiuserEnabled: true }
      )
    ).toEqual({ canManage: false, canViewDetails: false });
  });

  it('preserves single-user behavior', () => {
    expect(getQueueItemAccess(item, { currentUserId: null, isAdmin: false, multiuserEnabled: false }).canManage).toBe(
      true
    );
  });
});

describe('queue item row actions', () => {
  it('offers cancel to pending, waiting and running items only', () => {
    expect(isCancellableQueueItem({ status: 'pending' })).toBe(true);
    expect(isCancellableQueueItem({ status: 'waiting' })).toBe(true);
    expect(isCancellableQueueItem({ status: 'in_progress' })).toBe(true);
    expect(isCancellableQueueItem({ status: 'completed' })).toBe(false);
    expect(isCancellableQueueItem({ status: 'failed' })).toBe(false);
    expect(isCancellableQueueItem({ status: 'canceled' })).toBe(false);
  });

  it('offers retry to failed and canceled roots but not to workflow-call children', () => {
    expect(isRetryableQueueItem({ parentItemId: null, status: 'failed' })).toBe(true);
    expect(isRetryableQueueItem({ status: 'canceled' })).toBe(true);
    expect(isRetryableQueueItem({ parentItemId: 3, status: 'failed' })).toBe(false);
    expect(isRetryableQueueItem({ parentItemId: null, status: 'completed' })).toBe(false);
    expect(isRetryableQueueItem({ parentItemId: null, status: 'pending' })).toBe(false);
    expect(isRetryableQueueItem({ origin: buildUtilityQueueItemOrigin('thumbnail'), status: 'failed' })).toBe(false);
  });
});
