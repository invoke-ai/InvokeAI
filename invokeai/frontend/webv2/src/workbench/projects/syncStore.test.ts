import { accountLifecycle } from '@platform/state/accountLifecycle';
import { beforeEach, describe, expect, it } from 'vitest';

import {
  getProjectSyncSnapshot,
  reportProjectSync,
  reportProjectSyncEntry,
  resolveProjectSyncConflict,
} from './syncStore';

beforeEach(() => {
  accountLifecycle.activate('sync-store-test');
});

describe('project sync status', () => {
  it('preserves service-level pending state across single-project updates', () => {
    reportProjectSync({ hasPendingChanges: true, localDraftStatus: 'ok', projects: {}, recoverableDrafts: [] });

    reportProjectSyncEntry(
      'project-1',
      { isPendingPush: false, revision: 2 },
      { hasPendingChanges: true, localDraftStatus: 'ok', recoverableDrafts: [] }
    );
    resolveProjectSyncConflict('project-1', undefined, true);

    expect(getProjectSyncSnapshot().hasPendingChanges).toBe(true);
    expect(getProjectSyncSnapshot().lastSyncedAt).toBeNull();
  });
});
