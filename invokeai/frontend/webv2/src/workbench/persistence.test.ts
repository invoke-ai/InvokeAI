import { describe, expect, it } from 'vitest';

import { migrateWorkbenchPersistenceSnapshot } from './persistence';
import { createInitialWorkbenchState } from './workbenchState';

describe('workbench persistence migration', () => {
  it('accepts current versioned workbench snapshots', () => {
    const state = createInitialWorkbenchState();
    const snapshot = migrateWorkbenchPersistenceSnapshot({ savedAt: '2026-06-09T00:00:00.000Z', state, version: 1 });

    expect(snapshot).toEqual({ savedAt: '2026-06-09T00:00:00.000Z', state, version: 1 });
  });

  it('migrates legacy schemaVersion snapshots to the authoritative version field', () => {
    const state = createInitialWorkbenchState();
    const snapshot = migrateWorkbenchPersistenceSnapshot({
      savedAt: '2026-06-09T00:00:00.000Z',
      schemaVersion: 1,
      state,
    });

    expect(snapshot?.version).toBe(1);
    expect(snapshot?.state.projects).toHaveLength(3);
  });

  it('rejects unsupported persistence snapshots', () => {
    expect(migrateWorkbenchPersistenceSnapshot({ state: createInitialWorkbenchState(), version: 999 })).toBeNull();
    expect(migrateWorkbenchPersistenceSnapshot({ state: { projects: [] }, version: 1 })).toBeNull();
  });
});
