import { beforeEach, describe, expect, it, vi } from 'vitest';

import {
  hydratePersistedWorkbenchSnapshot,
  localStorageWorkbenchPersistence,
  serializeWorkbenchPersistenceSnapshot,
} from './persistence';
import { createInitialWorkbenchState, workbenchReducer } from './workbenchState.testing';

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
    const snapshot = hydratePersistedWorkbenchSnapshot({ savedAt: '2026-06-09T00:00:00.000Z', state, version: 1 });

    expect(snapshot).toEqual({ savedAt: '2026-06-09T00:00:00.000Z', state, version: 1 });
  });

  it('migrates legacy schemaVersion snapshots to the authoritative version field', () => {
    const state = createInitialWorkbenchState();
    const snapshot = hydratePersistedWorkbenchSnapshot({
      savedAt: '2026-06-09T00:00:00.000Z',
      schemaVersion: 1,
      state,
    });

    expect(snapshot?.version).toBe(1);
    expect(snapshot?.state.projects).toHaveLength(1);
  });

  it('drops legacy error logs from persisted snapshots', () => {
    const state = { ...createInitialWorkbenchState(), errorLog: ['old error'] };
    const snapshot = hydratePersistedWorkbenchSnapshot({ savedAt: '2026-06-09T00:00:00.000Z', state, version: 1 });

    expect(snapshot?.state).not.toHaveProperty('errorLog');
  });

  it('rejects unsupported persistence snapshots', () => {
    expect(hydratePersistedWorkbenchSnapshot({ state: createInitialWorkbenchState(), version: 999 })).toBeNull();
    expect(hydratePersistedWorkbenchSnapshot({ state: { projects: [] }, version: 1 })).toBeNull();
  });

  it('maps the trusted live snapshot to a versioned untrusted storage contract', () => {
    const state = createInitialWorkbenchState();
    const persisted = serializeWorkbenchPersistenceSnapshot({
      savedAt: '2026-06-09T00:00:00.000Z',
      state,
      version: 1,
    });

    expect(persisted).toEqual({ savedAt: '2026-06-09T00:00:00.000Z', state, version: 1 });
    expect(hydratePersistedWorkbenchSnapshot(persisted)?.state).toEqual(state);
  });

  it('drops corrupt localStorage snapshots instead of throwing', async () => {
    storage.set('invokeai:v7:webv2:workbench', '{not json');

    await expect(localStorageWorkbenchPersistence.loadWorkbench()).resolves.toBeNull();
    expect(storage.has('invokeai:v7:webv2:workbench')).toBe(false);
  });

  it('does not persist transient toast notifications', async () => {
    const state = workbenchReducer(createInitialWorkbenchState(), {
      kind: 'success',
      message: 'Old toast',
      title: 'Saved before reload',
      type: 'recordNotice',
    });

    const snapshot = await localStorageWorkbenchPersistence.saveWorkbench(state);

    expect(state.notifications).toHaveLength(1);
    expect(snapshot.state.notifications).toEqual([]);

    const raw = storage.get('invokeai:v7:webv2:workbench');
    const persisted = JSON.parse(raw ?? 'null') as { state: { notifications: unknown[] } };

    expect(persisted.state.notifications).toEqual([]);
  });

  it('persists empty project undo and redo stacks and normalizes legacy cache entries', async () => {
    let state = createInitialWorkbenchState();
    state = workbenchReducer(state, {
      sourceId: 'workflow',
      type: 'setInvocationSource',
    });
    expect(state.projects[0]?.undoRedo.past.length).toBeGreaterThan(0);

    const snapshot = await localStorageWorkbenchPersistence.saveWorkbench(state);
    expect(snapshot.state.projects[0]?.undoRedo).toEqual({ future: [], past: [] });

    const raw = storage.get('invokeai:v7:webv2:workbench');
    const hydrated = await localStorageWorkbenchPersistence.loadWorkbench();

    expect(raw).toBeDefined();
    expect(hydrated?.state.projects[0]?.undoRedo).toEqual({ future: [], past: [] });
  });

  it('treats localStorage quota failures as cache misses, not save failures', async () => {
    const originalSet = window.localStorage.setItem;

    try {
      window.localStorage.setItem = (key: string, value: string): void => {
        if (key === 'invokeai:v7:webv2:workbench') {
          throw new DOMException('Quota exceeded', 'QuotaExceededError');
        }

        originalSet.call(window.localStorage, key, value);
      };

      await expect(
        localStorageWorkbenchPersistence.saveWorkbench(createInitialWorkbenchState())
      ).resolves.toMatchObject({
        version: 1,
      });
    } finally {
      window.localStorage.setItem = originalSet;
    }
  });

  it('still attempts localStorage cache writes for large open workflow projects', async () => {
    const state = createInitialWorkbenchState();
    const project = state.projects[0]!;
    const largeState = {
      ...state,
      projects: [
        {
          ...project,
          projectGraph: {
            ...project.projectGraph,
            nodes: Array.from({ length: 300 }, (_, index) => ({
              data: { label: '', notes: '' },
              id: `node-${index}`,
              position: { x: 0, y: 0 },
              type: 'notes' as const,
            })),
          },
        },
      ],
    };

    await localStorageWorkbenchPersistence.saveWorkbench(largeState);

    expect(storage.has('invokeai:v7:webv2:workbench')).toBe(true);
  });
});
