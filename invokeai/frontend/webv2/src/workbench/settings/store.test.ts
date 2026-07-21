import { beforeEach, describe, expect, it, vi } from 'vitest';

import type * as storeModule from './store';

/**
 * The settings store's load/patch contract: backend-first with legacy
 * migration, resolved once per account scope, and offline edits replay
 * instead of being reverted by a stale server copy.
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

const seedBackendPreferences = (preferences: Record<string, unknown>): void => {
  api.__clientState.set(SETTINGS_KEY, JSON.stringify(preferences));
};

const readBackendPreferences = (): Record<string, unknown> =>
  JSON.parse(api.__clientState.get(SETTINGS_KEY) ?? 'null') as Record<string, unknown>;

beforeEach(async () => {
  vi.resetModules();
  api.__clientState.clear();
  api.deleteClientStateValue.mockClear();
  api.getClientStateValue.mockClear();
  api.setClientStateValue.mockClear();
  auth.scope = '';
  storage.clear();

  store = await import('./store');
});

describe('loadWorkbenchSettings', () => {
  it('adopts backend preferences without writing them back', async () => {
    seedBackendPreferences({ themeId: 'forest' });

    const preferences = await store.loadWorkbenchSettings();

    expect(preferences.themeId).toBe('forest');
    expect(api.setClientStateValue).not.toHaveBeenCalled();
    expect(store.useWorkbenchSettings).toBeDefined();
    expect(store.getWorkbenchPreferences().themeId).toBe('forest');
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

  it('resolves once per scope and reloads when the scope changes', async () => {
    seedBackendPreferences({ themeId: 'forest' });
    await store.loadWorkbenchSettings();
    await store.loadWorkbenchSettings();

    expect(api.getClientStateValue.mock.calls.filter(([key]) => key === SETTINGS_KEY)).toHaveLength(1);

    auth.scope = ':user:u2';
    seedBackendPreferences({ themeId: 'light' });

    const preferences = await store.loadWorkbenchSettings();

    expect(preferences.themeId).toBe('light');
  });

  it('heals invalid stored values to defaults', async () => {
    seedBackendPreferences({ developerLogLevel: 'shout', themeId: 'dark' });

    const preferences = await store.loadWorkbenchSettings();

    expect(preferences.themeId).toBe('classic');
    expect(preferences.developerLogLevel).toBe('warn');
  });

  it('serves the local copy when the backend is unreachable and retries next time', async () => {
    seedBackendPreferences({ themeId: 'forest' });
    await store.loadWorkbenchSettings();

    vi.resetModules();
    store = await import('./store');
    api.getClientStateValue.mockRejectedValueOnce(new Error('offline'));

    const offline = await store.loadWorkbenchSettings();

    expect(offline.themeId).toBe('forest');
    expect(store.useWorkbenchSettings).toBeDefined();

    const recovered = await store.loadWorkbenchSettings();

    expect(recovered.themeId).toBe('forest');
  });

  it('reports an error with defaults when nothing is available anywhere', async () => {
    api.getClientStateValue.mockRejectedValueOnce(new Error('offline'));

    const preferences = await store.loadWorkbenchSettings();

    expect(preferences.themeId).toBe('classic');
  });
});

describe('patchWorkbenchPreferences', () => {
  it('persists to the backend and localStorage', async () => {
    seedBackendPreferences({ themeId: 'forest' });
    await store.loadWorkbenchSettings();

    await store.patchWorkbenchPreferences({ reduceMotion: true });

    expect(readBackendPreferences().reduceMotion).toBe(true);
    expect(readBackendPreferences().themeId).toBe('forest');
    expect(store.getWorkbenchPreferences().reduceMotion).toBe(true);
    expect(store.getWorkbenchReduceMotion()).toBe(true);
  });

  it('persists account-bound custom hotkeys', async () => {
    seedBackendPreferences({ themeId: 'forest' });
    await store.loadWorkbenchSettings();

    await store.patchWorkbenchPreferences({ customHotkeys: { 'app.invoke': ['mod+shift+enter'] } });

    expect(readBackendPreferences().customHotkeys).toEqual({ 'app.invoke': ['mod+shift+enter'] });
    expect(store.getWorkbenchPreferences().customHotkeys).toEqual({ 'app.invoke': ['mod+shift+enter'] });
  });

  it('persists empty custom hotkeys as disabled bindings', async () => {
    await store.loadWorkbenchSettings();

    await store.patchWorkbenchPreferences({ customHotkeys: { 'app.invoke': [] } });

    expect(readBackendPreferences().customHotkeys).toEqual({ 'app.invoke': [] });
    expect(store.getWorkbenchPreferences().customHotkeys).toEqual({ 'app.invoke': [] });
  });

  it('replays an offline edit on the next load instead of reverting to the server copy', async () => {
    seedBackendPreferences({ themeId: 'forest' });
    await store.loadWorkbenchSettings();

    api.setClientStateValue.mockRejectedValueOnce(new Error('offline'));
    await store.patchWorkbenchPreferences({ themeId: 'light' });

    expect(readBackendPreferences().themeId).toBe('forest');

    vi.resetModules();
    store = await import('./store');

    const preferences = await store.loadWorkbenchSettings();

    expect(preferences.themeId).toBe('light');
    expect(readBackendPreferences().themeId).toBe('light');
  });
});

describe('clearWorkbenchSettings', () => {
  it('resets to defaults locally and deletes the backend key', async () => {
    seedBackendPreferences({ themeId: 'forest' });
    await store.loadWorkbenchSettings();

    await store.clearWorkbenchSettings();

    expect(store.getWorkbenchPreferences().themeId).toBe('classic');
    expect(api.__clientState.has(SETTINGS_KEY)).toBe(false);
  });
});

describe('normalizeProjectSettings', () => {
  it('defaults prompt syntax highlighting off for older project payloads', () => {
    expect(store.normalizeProjectSettings({}).showPromptSyntaxHighlighting).toBe(false);
    expect(store.normalizeProjectSettings({ showPromptSyntaxHighlighting: true }).showPromptSyntaxHighlighting).toBe(
      true
    );
    expect(store.normalizeProjectSettings({ showPromptSyntaxHighlighting: false }).showPromptSyntaxHighlighting).toBe(
      false
    );
  });
});

describe('normalizeWorkbenchPreferences diagnostics', () => {
  it('heals invalid diagnostics preferences to defaults', () => {
    expect(
      store.normalizeWorkbenchPreferences({
        developerLogEnabled: true,
        developerLogLevel: 'shout' as never,
        developerLogNamespaces: ['system', 'missing'] as never,
        developerPerformanceTimingsEnabled: true,
      })
    ).toMatchObject({
      developerLogEnabled: true,
      developerLogLevel: 'warn',
      developerLogNamespaces: ['system'],
      developerPerformanceTimingsEnabled: true,
    });
  });
});

describe('normalizeWorkbenchPreferences queue scope', () => {
  it('defaults queue jobs to all jobs for older preferences', () => {
    expect(store.normalizeWorkbenchPreferences({}).queueJobsScope).toBe('all');
  });

  it('accepts only supported queue job scopes', () => {
    expect(store.normalizeWorkbenchPreferences({ queueJobsScope: 'active-project' }).queueJobsScope).toBe(
      'active-project'
    );
    expect(store.normalizeWorkbenchPreferences({ queueJobsScope: 'all' }).queueJobsScope).toBe('all');
    expect(store.normalizeWorkbenchPreferences({ queueJobsScope: 'all-projects' }).queueJobsScope).toBe('all');
    expect(store.normalizeWorkbenchPreferences({ queueJobsScope: 'current-project' }).queueJobsScope).toBe('all');
  });
});

describe('normalizeWorkbenchPreferences generate sections', () => {
  it('defaults to no section overrides', () => {
    expect(store.normalizeWorkbenchPreferences({}).generateSectionsOpen).toEqual({});
  });

  it('keeps only boolean entries', () => {
    expect(
      store.normalizeWorkbenchPreferences({
        generateSectionsOpen: { advanced: true, concepts: false, model: 'yes' as never },
      }).generateSectionsOpen
    ).toEqual({ advanced: true, concepts: false });
  });

  it('heals malformed shapes to no overrides', () => {
    expect(
      store.normalizeWorkbenchPreferences({ generateSectionsOpen: ['advanced'] as never }).generateSectionsOpen
    ).toEqual({});
  });
});

describe('normalizeWorkbenchPreferences language', () => {
  it('accepts locale files that were migrated from the legacy web app', () => {
    expect(store.normalizeWorkbenchPreferences({ language: 'uk' }).language).toBe('uk');
    expect(store.normalizeWorkbenchPreferences({ language: 'en-GB' }).language).toBe('en-GB');
  });

  it('migrates legacy ua preferences to the uk locale file', () => {
    expect(store.normalizeWorkbenchPreferences({ language: 'ua' as never }).language).toBe('uk');
  });

  it('heals unsupported languages to the default', () => {
    expect(store.normalizeWorkbenchPreferences({ language: 'missing' as never }).language).toBe('en');
  });
});
