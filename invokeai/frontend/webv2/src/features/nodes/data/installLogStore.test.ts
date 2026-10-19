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
});
