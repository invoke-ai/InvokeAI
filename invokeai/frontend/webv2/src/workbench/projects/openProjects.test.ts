import { SESSION_STATE_KEY } from '@workbench/projects/session';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const getClientStateValue = vi.hoisted(() => vi.fn<(key: string, signal?: AbortSignal) => Promise<string | null>>());

vi.mock('@workbench/projects/api', () => ({ getClientStateValue }));

const importStore = () => {
  vi.resetModules();

  return import('@workbench/projects/openProjects');
};

describe('openProjects store', () => {
  beforeEach(() => {
    getClientStateValue.mockReset();
  });

  afterEach(() => {
    vi.resetModules();
  });

  it('reports a saved open set and the active project', async () => {
    getClientStateValue.mockResolvedValue(
      JSON.stringify({ account: {}, activeProjectId: 'b', openProjectIds: ['a', 'b'] })
    );

    const { getOpenProjects, refreshOpenProjects } = await importStore();

    await refreshOpenProjects();

    expect(getOpenProjects()).toEqual({ activeId: 'b', ids: ['a', 'b'], status: 'ready' });
    // The read is account-scoped, so it carries that scope's abort signal.
    expect(getClientStateValue).toHaveBeenCalledWith(SESSION_STATE_KEY, expect.any(AbortSignal));
  });

  it('distinguishes a definitely-empty session from an unknown one', async () => {
    getClientStateValue.mockResolvedValue(JSON.stringify({ account: {}, activeProjectId: '', openProjectIds: [] }));

    const empty = await importStore();

    await empty.refreshOpenProjects();
    expect(empty.getOpenProjects().ids).toEqual([]);

    // A blob written before the open set existed carries no `openProjectIds`.
    getClientStateValue.mockResolvedValue(JSON.stringify({ account: {}, activeProjectId: 'a' }));

    const legacy = await importStore();

    await legacy.refreshOpenProjects();
    expect(legacy.getOpenProjects().ids).toBeNull();
  });

  it('treats an unreachable backend as unknown rather than empty', async () => {
    getClientStateValue.mockRejectedValue(new Error('offline'));

    const { getOpenProjects, refreshOpenProjects } = await importStore();

    await refreshOpenProjects();

    expect(getOpenProjects()).toEqual({ activeId: null, ids: null, status: 'ready' });
  });

  it('shares one request between concurrent refreshes', async () => {
    getClientStateValue.mockResolvedValue(JSON.stringify({ account: {}, activeProjectId: 'a', openProjectIds: ['a'] }));

    const { refreshOpenProjects } = await importStore();

    await Promise.all([refreshOpenProjects(), refreshOpenProjects(), refreshOpenProjects()]);

    expect(getClientStateValue).toHaveBeenCalledTimes(1);
  });
});
