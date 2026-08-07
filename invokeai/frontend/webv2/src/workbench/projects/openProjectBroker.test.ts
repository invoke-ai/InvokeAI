import { beforeEach, describe, expect, it, vi } from 'vitest';

import { createOpenProjectBroker } from './openProjectBroker';
import { getOpenProject } from './syncStore';

/**
 * The registry has to say exactly which projects the editor holds: it is what decides whether a
 * library mutation goes through the sync engine or over HTTP, and a stale entry would put a write
 * on the wrong side of that.
 */

const createHarness = () => {
  const listeners = new Set<() => void>();
  const deps = {
    closeProject: vi.fn(),
    flushProject: vi.fn(() => Promise.resolve()),
    getOpenProjectIds: vi.fn(() => openIds),
    markProjectDeleted: vi.fn(),
    renameProject: vi.fn(),
    unmarkProjectDeleted: vi.fn(),
    subscribe: (listener: () => void) => {
      listeners.add(listener);

      return () => listeners.delete(listener);
    },
  };
  let openIds: string[] = [];

  return {
    deps,
    setOpenIds: (ids: string[]) => {
      openIds = ids;

      for (const listener of listeners) {
        listener();
      }
    },
  };
};

describe('createOpenProjectBroker', () => {
  beforeEach(async () => {
    const account = await import('@platform/state/accountLifecycle');

    account.accountLifecycle.activate('broker-user');
  });

  it('publishes a handle for every project the editor already holds', () => {
    const harness = createHarness();

    harness.setOpenIds(['a', 'b']);
    createOpenProjectBroker(harness.deps);

    expect(getOpenProject('a')).not.toBeNull();
    expect(getOpenProject('b')).not.toBeNull();
    expect(getOpenProject('c')).toBeNull();
  });

  it('follows the open set as tabs come and go', () => {
    const harness = createHarness();
    const broker = createOpenProjectBroker(harness.deps);

    harness.setOpenIds(['a']);
    expect(getOpenProject('a')).not.toBeNull();

    harness.setOpenIds(['b']);
    expect(getOpenProject('a')).toBeNull();
    expect(getOpenProject('b')).not.toBeNull();

    broker.dispose();
  });

  /** The handle is the mounted editor's. When the editor goes, every mutation must fall back to HTTP. */
  it('withdraws every handle when the editor unmounts', () => {
    const harness = createHarness();
    const broker = createOpenProjectBroker(harness.deps);

    harness.setOpenIds(['a', 'b']);
    broker.dispose();

    expect(getOpenProject('a')).toBeNull();
    expect(getOpenProject('b')).toBeNull();

    // And it stops listening: a later change must not resurrect them.
    harness.setOpenIds(['a']);
    expect(getOpenProject('a')).toBeNull();
  });

  it('renames through the reducer and then flushes, in that order', async () => {
    const harness = createHarness();
    const order: string[] = [];

    harness.deps.renameProject.mockImplementation(() => order.push('rename'));
    harness.deps.flushProject.mockImplementation(() => {
      order.push('flush');

      return Promise.resolve();
    });
    createOpenProjectBroker(harness.deps);
    harness.setOpenIds(['a']);

    await getOpenProject('a')!.rename('New name');

    expect(harness.deps.renameProject).toHaveBeenCalledWith('a', 'New name');
    expect(order).toEqual(['rename', 'flush']);
  });

  it('routes deletion and closing at the project it was published for', () => {
    const harness = createHarness();

    createOpenProjectBroker(harness.deps);
    harness.setOpenIds(['a']);

    const handle = getOpenProject('a')!;

    handle.markDeleted();
    handle.close();

    expect(harness.deps.markProjectDeleted).toHaveBeenCalledWith('a');
    expect(harness.deps.closeProject).toHaveBeenCalledWith('a');
  });
});
