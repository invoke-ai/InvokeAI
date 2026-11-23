import type * as accountLifecycleModule from '@platform/state/accountLifecycle';

import { beforeEach, describe, expect, it, vi } from 'vitest';

import type * as storeModule from './store';

/**
 * Verify backend-first loading, legacy migration, per-account resolution, and offline-edit replay over stale server
 * state.
 */

const api = vi.hoisted(() => {
  const clientState = new Map<string, string>();

  return {
    __clientState: clientState,
    deleteClientStateValue: vi.fn((key: string) => {
      clientState.delete(key);

      return Promise.resolve();
    }),
    getClientStateValue: vi.fn((key: string) => Promise.resolve(clientState.get(key) ?? null)),
    setClientStateValue: vi.fn((key: string, value: string) => {
      clientState.set(key, value);

      return Promise.resolve();
    }),
  };
});

const auth = vi.hoisted(() => ({ scope: '' }));

vi.mock('../projects/api', () => api);
vi.mock('@features/identity', () => ({ getUserStorageScope: () => auth.scope }));

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

const SETTINGS_KEY = 'webv2:workbench-settings';
const SESSION_KEY = 'webv2:workbench-account';

let store: typeof storeModule;
let account: typeof accountLifecycleModule;

const seedBackendPreferences = (preferences: Record<string, unknown>): void => {
  api.__clientState.set(SETTINGS_KEY, JSON.stringify(preferences));
};

const readBackendPreferences = (): Record<string, unknown> =>
  JSON.parse(api.__clientState.get(SETTINGS_KEY) ?? 'null') as Record<string, unknown>;

beforeEach(async () => {
  vi.resetModules();
  api.__clientState.clear();
  api.deleteClientStateValue.mockReset();
  api.deleteClientStateValue.mockImplementation((key: string) => {
    api.__clientState.delete(key);

    return Promise.resolve();
  });
  api.getClientStateValue.mockReset();
  api.getClientStateValue.mockImplementation((key: string) => Promise.resolve(api.__clientState.get(key) ?? null));
  api.setClientStateValue.mockReset();
  api.setClientStateValue.mockImplementation((key: string, value: string) => {
    api.__clientState.set(key, value);

    return Promise.resolve();
  });
  auth.scope = '';
  storage.clear();

  store = await import('./store');
  account = await import('@platform/state/accountLifecycle');
});

describe('loadWorkbenchSettings', () => {
  it('migrates a preference stored under a renamed theme id', async () => {
    seedBackendPreferences({ themeId: 'forest' });

    const preferences = await store.loadWorkbenchSettings();

    expect(preferences.themeId).toBe('osakaJade');
  });

  it('migrates legacy session-blob preferences once when the settings key is missing', async () => {
    api.__clientState.set(
      SESSION_KEY,
      JSON.stringify({
        account: { activeLayoutPresetId: 'gallery', preferences: { themeId: 'mono' } },
        activeProjectId: 'p1',
      })
    );

    const preferences = await store.loadWorkbenchSettings();

    expect(preferences.themeId).toBe('mono');
    expect(readBackendPreferences().themeId).toBe('mono');
    expect(api.setClientStateValue).toHaveBeenCalledTimes(1);
  });

  it('serves the local copy when the backend is unreachable and retries next time', async () => {
    seedBackendPreferences({ themeId: 'osakaJade' });
    await store.loadWorkbenchSettings();

    vi.resetModules();
    store = await import('./store');
    api.getClientStateValue.mockRejectedValueOnce(new Error('offline'));

    const offline = await store.loadWorkbenchSettings();

    expect(offline.themeId).toBe('osakaJade');
    expect(store.useWorkbenchSettings).toBeDefined();

    const recovered = await store.loadWorkbenchSettings();

    expect(recovered.themeId).toBe('osakaJade');
  });

  it('does not share a same-user single-flight or publish a prior epoch response', async () => {
    account.accountLifecycle.activate('user-a', ':user:a');
    let resolveOld: ((value: string | null) => void) | undefined;
    api.getClientStateValue
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveOld = resolve;
          })
      )
      .mockResolvedValueOnce(JSON.stringify({ themeId: 'light' }));
    const oldLoad = store.loadWorkbenchSettings();

    account.accountLifecycle.invalidate();
    account.accountLifecycle.activate('user-a', ':user:a');
    await expect(store.loadWorkbenchSettings()).resolves.toMatchObject({ themeId: 'light' });

    resolveOld?.(JSON.stringify({ themeId: 'osakaJade' }));
    await expect(oldLoad).rejects.toThrow('no longer active');
    expect(store.getWorkbenchPreferences().themeId).toBe('light');
  });
});

describe('patchWorkbenchPreferences', () => {
  it('persists to the backend and localStorage', async () => {
    seedBackendPreferences({ themeId: 'osakaJade' });
    await store.loadWorkbenchSettings();

    await store.patchWorkbenchPreferences({ reduceMotion: true });

    expect(readBackendPreferences().reduceMotion).toBe(true);
    expect(readBackendPreferences().themeId).toBe('osakaJade');
    expect(store.getWorkbenchPreferences().reduceMotion).toBe(true);
    expect(store.getWorkbenchReduceMotion()).toBe(true);
  });

  it('replays an offline edit on the next load instead of reverting to the server copy', async () => {
    seedBackendPreferences({ themeId: 'osakaJade' });
    await store.loadWorkbenchSettings();

    api.setClientStateValue.mockRejectedValueOnce(new Error('offline'));
    await store.patchWorkbenchPreferences({ themeId: 'light' });

    expect(readBackendPreferences().themeId).toBe('osakaJade');

    vi.resetModules();
    store = await import('./store');

    const preferences = await store.loadWorkbenchSettings();

    expect(preferences.themeId).toBe('light');
    expect(readBackendPreferences().themeId).toBe('light');
  });

  it('serializes writes so an older completion cannot overwrite a newer preference snapshot', async () => {
    await store.loadWorkbenchSettings();
    api.setClientStateValue.mockClear();
    let resolveFirstWrite: (() => void) | undefined;
    api.setClientStateValue.mockImplementationOnce(
      (key: string, value: string) =>
        new Promise<void>((resolve) => {
          resolveFirstWrite = () => {
            api.__clientState.set(key, value);
            resolve();
          };
        })
    );

    const first = store.patchWorkbenchPreferences({ themeId: 'osakaJade' });
    const second = store.patchWorkbenchPreferences({ themeId: 'light' });

    await vi.waitFor(() => {
      expect(api.setClientStateValue).toHaveBeenCalledTimes(1);
    });
    resolveFirstWrite?.();
    await Promise.all([first, second]);

    expect(readBackendPreferences().themeId).toBe('light');
    expect(store.getWorkbenchPreferences().themeId).toBe('light');
  });

  it('settles without publishing an error when its account expires during a write', async () => {
    account.accountLifecycle.activate('user-a', ':user:a');
    await store.loadWorkbenchSettings();
    api.setClientStateValue.mockClear();
    let resolveWrite: (() => void) | undefined;
    api.setClientStateValue.mockImplementationOnce(
      (key: string, value: string) =>
        new Promise<void>((resolve) => {
          resolveWrite = () => {
            api.__clientState.set(key, value);
            resolve();
          };
        })
    );

    const write = store.patchWorkbenchPreferences({ themeId: 'osakaJade' });
    account.accountLifecycle.invalidate();
    account.accountLifecycle.activate('user-b', ':user:b');
    resolveWrite?.();

    await expect(write).resolves.toBeUndefined();
    expect(store.getWorkbenchPreferences().themeId).toBe('classic');
  });
});

describe('clearWorkbenchSettings', () => {
  it('resets to defaults locally and deletes the backend key', async () => {
    seedBackendPreferences({ themeId: 'osakaJade' });
    await store.loadWorkbenchSettings();

    await store.clearWorkbenchSettings();

    expect(store.getWorkbenchPreferences().themeId).toBe('classic');
    expect(api.__clientState.has(SETTINGS_KEY)).toBe(false);
  });

  it('orders reset after an in-flight patch so the older write cannot restore backend or local settings', async () => {
    await store.loadWorkbenchSettings();
    api.setClientStateValue.mockClear();
    api.deleteClientStateValue.mockClear();
    let resolvePatch: (() => void) | undefined;
    api.setClientStateValue.mockImplementationOnce(
      (key: string, value: string) =>
        new Promise<void>((resolve) => {
          resolvePatch = () => {
            api.__clientState.set(key, value);
            resolve();
          };
        })
    );

    const patch = store.patchWorkbenchPreferences({ themeId: 'osakaJade' });
    const clear = store.clearWorkbenchSettings();

    await vi.waitFor(() => {
      expect(api.setClientStateValue).toHaveBeenCalledTimes(1);
    });
    expect(api.deleteClientStateValue).not.toHaveBeenCalled();
    resolvePatch?.();
    await Promise.all([patch, clear]);

    expect(api.__clientState.has(SETTINGS_KEY)).toBe(false);
    expect(storage.has('invokeai:v7:webv2:settings')).toBe(false);
    expect(store.getWorkbenchPreferences().themeId).toBe('classic');
  });
});

describe('normalizeWorkbenchPreferences boolean preferences', () => {
  const read = (key: keyof typeof storeModule.DEFAULT_PREFERENCES, stored?: unknown): unknown =>
    store.normalizeWorkbenchPreferences(stored === undefined ? {} : ({ [key]: stored } as never))[key];

  it.each([
    ['showPromptSyntaxHighlighting', true, 'no'],
    ['preferNumericAttentionStyle', false, 'yes'],
    ['highContrast', false, 1],
    ['selectObjectAutoUpdate', true, 'false'],
    ['installInPlace', true, 'false'],
    ['workflowEdgesBehindNodes', false, 'yes'],
    ['autoSwitchInvocationRoute', true, 'off'],
    ['antialiasProgressImages', false, 'yes'],
  ] as const)('keeps %s at %s unless a boolean is stored', (key, fallback, nonBoolean) => {
    expect(read(key)).toBe(fallback);
    expect(read(key, !fallback)).toBe(!fallback);
    expect(read(key, fallback)).toBe(fallback);
    expect(read(key, nonBoolean)).toBe(fallback);
  });
});

describe('normalizeWorkbenchPreferences workflow library', () => {
  it('gives a key stored before directions existed its natural direction', () => {
    expect(store.normalizeWorkbenchPreferences({ workflowLibrarySort: 'name' })).toMatchObject({
      workflowLibrarySort: 'name',
      workflowLibrarySortDirection: 'asc',
    });
    expect(store.normalizeWorkbenchPreferences({ workflowLibrarySort: 'updated' })).toMatchObject({
      workflowLibrarySort: 'updated',
      workflowLibrarySortDirection: 'desc',
    });
    expect(store.normalizeWorkbenchPreferences({ workflowLibrarySort: 'opened' })).toMatchObject({
      workflowLibrarySort: 'opened',
      workflowLibrarySortDirection: 'desc',
    });
    // A stored direction wins.
    expect(
      store.normalizeWorkbenchPreferences({ workflowLibrarySort: 'name', workflowLibrarySortDirection: 'desc' })
    ).toMatchObject({ workflowLibrarySortDirection: 'desc' });
  });
});

describe('Select Object auto-update preference', () => {
  it('starts on for an account whose saves wrote the old auto-process field as false', async () => {
    seedBackendPreferences({ selectObjectAutoProcess: false, themeId: 'osakaJade' });

    expect((await store.loadWorkbenchSettings()).selectObjectAutoUpdate).toBe(true);
    await store.patchWorkbenchPreferences({ reduceMotion: true });

    expect(readBackendPreferences()).not.toHaveProperty('selectObjectAutoProcess');
    expect(readBackendPreferences().selectObjectAutoUpdate).toBe(true);
  });
});

describe('normalizeWorkbenchPreferences queue scope', () => {
  it('treats a dismissed alpha notice as a dismissed beta notice', () => {
    expect(store.normalizeWorkbenchPreferences({ alphaNoticeAcknowledged: true }).betaNoticeAcknowledged).toBe(true);
    expect(store.normalizeWorkbenchPreferences({}).betaNoticeAcknowledged).toBe(false);
  });
});

describe('normalizeWorkbenchPreferences language', () => {
  it('migrates legacy ua preferences to the uk locale file', () => {
    expect(store.normalizeWorkbenchPreferences({ language: 'ua' as never }).language).toBe('uk');
  });
});

describe('controlModelMemory preference', () => {
  it('keeps the newest 256 picks', () => {
    const memory = Object.fromEntries(Array.from({ length: 300 }, (_, index) => [`base-${index}:depth`, `m${index}`]));
    const kept = store.normalizeWorkbenchPreferences({ controlModelMemory: memory }).controlModelMemory;

    expect(Object.keys(kept)).toHaveLength(256);
    expect(kept['base-299:depth']).toBe('m299');
    expect(kept['base-43:depth']).toBeUndefined();
  });

  it('remembers, replaces and forgets a pick through the persisted preferences, and resets with the rest', async () => {
    await store.loadWorkbenchSettings();

    await store.patchControlModelMemory('sdxl:depth', 'depth-a');
    await store.patchControlModelMemory('flux:pose', 'flux-union');
    await store.patchControlModelMemory('sdxl:depth', 'depth-b');

    expect(readBackendPreferences().controlModelMemory).toEqual({ 'flux:pose': 'flux-union', 'sdxl:depth': 'depth-b' });

    await store.patchControlModelMemory('flux:pose', null);

    expect(store.getWorkbenchPreferences().controlModelMemory).toEqual({ 'sdxl:depth': 'depth-b' });
    expect(readBackendPreferences().controlModelMemory).toEqual({ 'sdxl:depth': 'depth-b' });

    await store.clearWorkbenchSettings();

    expect(store.getWorkbenchPreferences().controlModelMemory).toEqual({});
  });
});

describe('scopedDefaults preference', () => {
  it('sets and clears one owner key at a time, keeping the others, and persists them', async () => {
    await store.loadWorkbenchSettings();

    await store.patchScopedDefault('gallery', { thumbnailFit: 'aspect' });
    await store.patchScopedDefault('gallery', { showDateBoards: true });
    await store.patchScopedDefault('generate', { useWatermark: true });

    expect(store.getWorkbenchPreferences().scopedDefaults).toEqual({
      gallery: { showDateBoards: true, thumbnailFit: 'aspect' },
      generate: { useWatermark: true },
    });
    expect(readBackendPreferences().scopedDefaults).toEqual(store.getWorkbenchPreferences().scopedDefaults);

    await store.clearScopedDefault('gallery', ['thumbnailFit']);
    await store.clearScopedDefault('generate', ['useWatermark']);

    expect(store.getWorkbenchPreferences().scopedDefaults).toEqual({ gallery: { showDateBoards: true } });
  });
});

describe('aspectRatioPresets preference', () => {
  const cinema = { height: 1, id: 'cinema', label: 'Cinema', width: 2.39 };

  it('drops malformed entries and repeated ids but leaves ratio rules to generation', () => {
    // 20:1 is outside this build's range; a newer build may allow it, so storage keeps it for that build.
    const wide = { height: 1, id: 'wide', label: 'Wide', width: 20 };

    expect(
      store.normalizeWorkbenchPreferences({
        aspectRatioPresets: [
          cinema,
          null,
          { height: 1, id: 'no-label', label: ' ', width: 2 },
          { height: '1', id: 'text', label: 'Text', width: 2 },
          { ...cinema, label: 'Repeat' },
          wide,
        ] as never,
      }).aspectRatioPresets
    ).toEqual([cinema, wide]);
  });
});
