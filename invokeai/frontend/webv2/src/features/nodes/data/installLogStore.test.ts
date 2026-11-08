import { beforeEach, describe, expect, it, vi } from 'vitest';

describe('custom node install log store', () => {
  beforeEach(() => {
    vi.resetModules();
  });

  it('resolves an entry in place so no orphaned installing rows remain', async () => {
    const store = await import('./installLogStore');

    const entry = store.addCustomNodeInstallLogEntry({ name: 'https://x/pack.git', status: 'installing' });

    store.updateCustomNodeInstallLogEntry(entry.id, { message: 'ok', name: 'pack', status: 'completed' });

    const log = store.getCustomNodeInstallLogForTests();

    expect(log).toHaveLength(1);
    expect(log[0]).toMatchObject({ id: entry.id, message: 'ok', name: 'pack', status: 'completed' });
  });

  it('leaves other entries untouched and ignores unknown ids', async () => {
    const store = await import('./installLogStore');

    const first = store.addCustomNodeInstallLogEntry({ name: 'one', status: 'installing' });
    const second = store.addCustomNodeInstallLogEntry({ name: 'two', status: 'installing' });

    store.updateCustomNodeInstallLogEntry(second.id, { status: 'error' });
    store.updateCustomNodeInstallLogEntry(9999, { status: 'completed' });

    const log = store.getCustomNodeInstallLogForTests();

    expect(log.find((entry) => entry.id === first.id)?.status).toBe('installing');
    expect(log.find((entry) => entry.id === second.id)?.status).toBe('error');
  });

  it('clears settled entries but keeps an install in flight so its outcome can still land', async () => {
    const store = await import('./installLogStore');

    const running = store.addCustomNodeInstallLogEntry({
      name: 'running',
      source: 'https://x/running.git',
      status: 'installing',
    });
    store.addCustomNodeInstallLogEntry({ name: 'done', status: 'completed' });
    store.addCustomNodeInstallLogEntry({ name: 'failed', status: 'error' });
    store.addCustomNodeInstallLogEntry({ name: 'gone', status: 'uninstalled' });

    store.clearFinishedCustomNodeInstalls();
    store.updateCustomNodeInstallLogEntry(running.id, { name: 'running-pack', status: 'completed' });

    expect(store.getCustomNodeInstallLogForTests()).toEqual([
      expect.objectContaining({
        id: running.id,
        name: 'running-pack',
        source: 'https://x/running.git',
        status: 'completed',
      }),
    ]);
  });

  it('removes only the named entry', async () => {
    const store = await import('./installLogStore');

    const kept = store.addCustomNodeInstallLogEntry({ name: 'kept', status: 'completed' });
    const removed = store.addCustomNodeInstallLogEntry({ name: 'removed', status: 'error' });

    store.removeCustomNodeInstallLogEntry(removed.id);

    expect(store.getCustomNodeInstallLogForTests().map((entry) => entry.id)).toEqual([kept.id]);
  });
});
