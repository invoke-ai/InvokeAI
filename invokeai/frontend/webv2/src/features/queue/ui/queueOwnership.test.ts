import type { QueueItemReadModel } from '@features/queue/core/types';

import { describe, expect, it } from 'vitest';

import { getQueueItemAccess } from './queueOwnership';

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
