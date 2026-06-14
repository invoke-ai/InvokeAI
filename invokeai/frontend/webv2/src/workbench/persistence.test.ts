import { beforeEach, describe, expect, it, vi } from 'vitest';

import { localStorageWorkbenchPersistence, migrateWorkbenchPersistenceSnapshot } from './persistence';
import { createInitialWorkbenchState } from './workbenchState';

const storage = new Map<string, string>();

vi.stubGlobal('window', {
  localStorage: {
    getItem: (key: string): string | null => storage.get(key) ?? null,
    removeItem: (key: string): void => {
      storage.delete(key);
    },
    setItem: (key: string, value: string): void => {
      storage.set(key, value);
    },
  },
});

beforeEach(() => {
  storage.clear();
});

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
    expect(snapshot?.state.projects).toHaveLength(1);
  });

  it('rejects unsupported persistence snapshots', () => {
    expect(migrateWorkbenchPersistenceSnapshot({ state: createInitialWorkbenchState(), version: 999 })).toBeNull();
    expect(migrateWorkbenchPersistenceSnapshot({ state: { projects: [] }, version: 1 })).toBeNull();
  });

  it('drops corrupt localStorage snapshots instead of throwing', async () => {
    storage.set('invokeai:v7:webv2:workbench', '{not json');

    await expect(localStorageWorkbenchPersistence.loadWorkbench()).resolves.toBeNull();
    expect(storage.has('invokeai:v7:webv2:workbench')).toBe(false);
  });
});
