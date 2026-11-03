import { describe, expect, it, vi } from 'vitest';

import { createQueueRunLockPort, getQueueRunLockName } from './queueRunLocks';

describe('queue run locks', () => {
  it('isolates ownership by account, project, and queue item without delimiter collisions', async () => {
    const acquire = vi.fn().mockResolvedValue({ kind: 'contended' });
    const acquireProject = vi.fn().mockResolvedValue({ kind: 'acquired', release: vi.fn() });
    const locks = createQueueRunLockPort(':account-1', acquire, acquireProject);

    await expect(locks.acquire('a:b', 'c')).resolves.toEqual({ kind: 'contended' });
    await locks.acquire('a', 'b:c');

    expect(acquire.mock.calls.map(([name]) => name)).toEqual([
      getQueueRunLockName(':account-1', 'a:b', 'c'),
      getQueueRunLockName(':account-1', 'a', 'b:c'),
    ]);
    expect(acquire.mock.calls[0]?.[0]).not.toBe(acquire.mock.calls[1]?.[0]);
  });

  it('asks whether a run is held under the name its owner acquires', async () => {
    const acquire = vi.fn().mockResolvedValue({ kind: 'contended' });
    const isHeld = vi.fn().mockResolvedValue(true);
    const locks = createQueueRunLockPort(
      ':account-1',
      acquire,
      vi.fn().mockResolvedValue({ kind: 'acquired', release: vi.fn() }),
      isHeld
    );

    await locks.acquire('project-1', 'item-1');
    await expect(locks.isHeld('project-1', 'item-1')).resolves.toBe(true);

    expect(isHeld).toHaveBeenCalledWith(acquire.mock.calls[0]?.[0]);
  });

  it('waits on a run under the name its owner acquires, taking it only for a moment', async () => {
    const acquire = vi.fn().mockResolvedValue({ kind: 'contended' });
    const release = vi.fn().mockResolvedValue(undefined);
    const waitShared = vi.fn().mockResolvedValue({ kind: 'acquired', release });
    const locks = createQueueRunLockPort(
      ':account-1',
      acquire,
      vi.fn().mockResolvedValue({ kind: 'acquired', release: vi.fn() }),
      vi.fn(),
      waitShared
    );
    const signal = new AbortController().signal;

    await locks.acquire('project-1', 'item-1');
    await locks.waitForRunRelease('project-1', 'item-1', signal);

    expect(waitShared).toHaveBeenCalledWith(acquire.mock.calls[0]?.[0], signal);
    // Released at once, so the tab that wants the run next is not kept out.
    expect(release).toHaveBeenCalledOnce();
  });

  it('holds a shared project lease for the lifetime of an acquired run', async () => {
    const calls: string[] = [];
    const locks = createQueueRunLockPort(
      ':account-1',
      vi.fn().mockResolvedValue({ kind: 'acquired', release: () => void calls.push('run') }),
      vi.fn().mockResolvedValue({ kind: 'acquired', release: () => void calls.push('project') })
    );

    const lock = await locks.acquire('project-1', 'item-1');
    expect(lock.kind).toBe('acquired');
    if (lock.kind === 'acquired') {
      await lock.release();
    }
    expect(calls).toEqual(['run', 'project']);
  });

  it('releases the project lease when run acquisition rejects', async () => {
    const releaseProject = vi.fn().mockResolvedValue(undefined);
    const locks = createQueueRunLockPort(
      ':account-1',
      vi.fn().mockRejectedValue(new Error('lock failure')),
      vi.fn().mockResolvedValue({ kind: 'acquired', release: releaseProject })
    );

    await expect(locks.acquire('project-1', 'item-1')).rejects.toThrow('lock failure');
    expect(releaseProject).toHaveBeenCalledOnce();
  });

  it('releases the project lease when run release rejects', async () => {
    const releaseProject = vi.fn().mockResolvedValue(undefined);
    const locks = createQueueRunLockPort(
      ':account-1',
      vi.fn().mockResolvedValue({ kind: 'acquired', release: () => Promise.reject(new Error('release failure')) }),
      vi.fn().mockResolvedValue({ kind: 'acquired', release: releaseProject })
    );

    const lock = await locks.acquire('project-1', 'item-1');
    if (lock.kind !== 'acquired') {
      throw new Error('Expected lock acquisition');
    }
    await expect(lock.release()).rejects.toThrow('release failure');
    expect(releaseProject).toHaveBeenCalledOnce();
  });
});
