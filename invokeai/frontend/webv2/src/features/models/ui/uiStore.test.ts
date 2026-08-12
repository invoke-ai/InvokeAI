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
