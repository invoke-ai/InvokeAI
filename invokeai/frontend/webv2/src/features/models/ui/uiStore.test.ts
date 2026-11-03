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

    // Clear prior scan, repo, and bundle results so the requested starter catalog remains visible.
    expect(snapshot.activeTab).toBe('add');
    expect(snapshot.addModelsQuery).toBe('Juggernaut XL');
    expect(snapshot.hfLookup).toBeNull();
    expect(snapshot.scan).toBeNull();
    expect(snapshot.selectedBundleName).toBeNull();
  });

  it('keeps the Add Models box text until a search request replaces it or the account changes', async () => {
    const store = await import('./uiStore');
    const { accountLifecycle } = await import('@platform/state/accountLifecycle');
    accountLifecycle.activate('account-a');
    const query = () => store.getModelsUiSnapshotForTests().addModelsQuery;

    expect(query()).toBe('');
    store.setAddModelsQuery('/home/me/loras');
    store.openModelManagerTab('keys');
    expect(query()).toBe('/home/me/loras');

    store.requestAddModelsSearch('Juggernaut XL');
    expect(query()).toBe('Juggernaut XL');

    accountLifecycle.activate('account-b');
    expect(query()).toBe('');
  });

  it('hands the starter type-filter seed over exactly once, clearing a search that would hide it', async () => {
    const store = await import('./uiStore');
    const query = () => store.getModelsUiSnapshotForTests().addModelsQuery;

    store.requestAddModelsSearch('Juggernaut XL');
    store.requestAddModelsTypeFilter('text_llm');
    expect(store.getModelsUiSnapshotForTests().activeTab).toBe('add');
    expect(store.getAddModelsTypeSeed()).toBe('text_llm');
    expect(query()).toBe('');

    store.requestAddModelsSearch('Juggernaut XL');
    expect(store.getAddModelsTypeSeed()).toBeNull();

    store.requestAddModelsTypeFilter('llava_onevision');
    store.clearAddModelsTypeSeed();
    expect(store.getAddModelsTypeSeed()).toBeNull();
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
  // Which pane a single-pane manager shows follows every opener and clear, without callers naming it.
  it('leaves the pane choice open until the user opens or leaves the detail', async () => {
    const store = await import('./uiStore');
    const detailOpen = () => store.getModelsUiSnapshotForTests().detailOpen;

    expect(detailOpen()).toBeNull();
    store.openModelDetail('a');
    expect(detailOpen()).toBe(true);
    store.closeModelDetail();
    expect(detailOpen()).toBe(false);
    // Re-opening the same model after Back reveals it again, as do tabs requested from elsewhere.
    store.openModelDetail('a');
    expect(detailOpen()).toBe(true);
    store.closeModelDetail();
    store.requestAddModelsSearch('flux');
    expect(detailOpen()).toBe(true);
  });

  it('returns to the library only when a delete removes the model on screen', async () => {
    const store = await import('./uiStore');
    const detailOpen = () => store.getModelsUiSnapshotForTests().detailOpen;

    store.openModelDetail('a');
    store.pruneModelsUiKeys(['b']);
    expect(detailOpen()).toBe(true);
    store.pruneModelsUiKeys(['a']);
    expect(detailOpen()).toBe(false);

    store.openModelDetail('c');
    store.openModelManagerTab('add');
    store.pruneModelsUiKeys(['c']);
    expect(detailOpen()).toBe(true);
  });
});
