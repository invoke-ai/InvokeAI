import { beforeEach, describe, expect, it, vi } from 'vitest';

describe('models ui store', () => {
  beforeEach(() => {
    vi.resetModules();
  });

  it('opens Add Models with a bundle preselected and clears stale results', async () => {
    const store = await import('./uiStore');

    store.updateModelsUi({
      activeTab: 'keys',
      hfLookup: { repo: 'owner/repo', urls: ['a'] },
      scan: { path: '/models', results: [] },
    });

    store.openAddModelsWithBundle('Essentials');

    const snapshot = store.getModelsUiSnapshotForTests();
    expect(snapshot.activeTab).toBe('add');
    expect(snapshot.selectedBundleName).toBe('Essentials');
    expect(snapshot.hfLookup).toBeNull();
    expect(snapshot.scan).toBeNull();
  });

  it('opens Add Models searching for one model, with nothing else in the way', async () => {
    const store = await import('./uiStore');

    store.updateModelsUi({
      activeTab: 'details',
      hfLookup: { repo: 'owner/repo', urls: ['a'] },
      scan: { path: '/models', results: [] },
      selectedBundleName: 'Essentials',
    });

    store.requestAddModelsSearch('Juggernaut XL');

    const snapshot = store.getModelsUiSnapshotForTests();

    // Add Models, showing the starter catalog filtered to the request — a
    // leftover scan/HF panel or bundle would hide the catalog entirely.
    expect(snapshot.activeTab).toBe('add');
    expect(store.getAddModelsSeed()).toBe('Juggernaut XL');
    expect(snapshot.hfLookup).toBeNull();
    expect(snapshot.scan).toBeNull();
    expect(snapshot.selectedBundleName).toBeNull();
  });

  it('hands the Add Models seed over exactly once', async () => {
    const store = await import('./uiStore');

    expect(store.getAddModelsSeed()).toBe('');

    store.requestAddModelsSearch('Juggernaut XL');

    // Reading is pure — StrictMode double-invokes the initializer that reads it.
    expect(store.getAddModelsSeed()).toBe('Juggernaut XL');
    expect(store.getAddModelsSeed()).toBe('Juggernaut XL');

    store.clearAddModelsSeed();

    // The next time Add Models opens on its own, the box is empty again.
    expect(store.getAddModelsSeed()).toBe('');
  });

  it('keeps the bundle selection until it is explicitly replaced', async () => {
    const store = await import('./uiStore');

    store.openAddModelsWithBundle('Essentials');
    store.openModelManagerTab('details');
    expect(store.getModelsUiSnapshotForTests().selectedBundleName).toBe('Essentials');

    store.updateModelsUi({ selectedBundleName: null });
    expect(store.getModelsUiSnapshotForTests().selectedBundleName).toBeNull();
  });

  it('prunes deleted keys from selection and the active slot', async () => {
    const store = await import('./uiStore');

    store.updateModelsUi({ activeModelKey: 'a', selectedKeys: new Set(['a', 'b']) });
    store.pruneModelsUiKeys(['a']);

    const snapshot = store.getModelsUiSnapshotForTests();
    expect(snapshot.activeModelKey).toBeNull();
    expect([...snapshot.selectedKeys]).toEqual(['b']);
  });
});
