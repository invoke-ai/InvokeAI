import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { ProjectPushOutcome } from './projectFlush';

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
    deleteProject: vi.fn(() => Promise.resolve()),
    flushProject: vi.fn(() => Promise.resolve<ProjectPushOutcome>({ documentJson: '{}', kind: 'acknowledged' })),
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
  const brokers: { dispose: () => void }[] = [];

  const startBroker = (deps: Parameters<typeof createOpenProjectBroker>[0]) => {
    const broker = createOpenProjectBroker(deps);

    brokers.push(broker);

    return broker;
  };

  beforeEach(async () => {
    const account = await import('@platform/state/accountLifecycle');

    account.accountLifecycle.activate('broker-user');
  });

  // The registry is module state. A test that leaves a handle in it is a test the next one
  // inherits, and these only pass in isolation by using different ids.
  afterEach(() => {
    for (const broker of brokers.splice(0)) {
      broker.dispose();
    }
  });

  it('publishes a handle for every project the editor already holds', () => {
    const harness = createHarness();

    harness.setOpenIds(['a', 'b']);
    startBroker(harness.deps);

    expect(getOpenProject('a')).not.toBeNull();
    expect(getOpenProject('b')).not.toBeNull();
    expect(getOpenProject('c')).toBeNull();
  });

  it('follows the open set as tabs come and go', () => {
    const harness = createHarness();
    const broker = startBroker(harness.deps);

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
    const broker = startBroker(harness.deps);

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

      return Promise.resolve<ProjectPushOutcome>({ documentJson: '{}', kind: 'acknowledged' });
    });
    startBroker(harness.deps);
    harness.setOpenIds(['a']);

    await getOpenProject('a')!.rename('New name');

    expect(harness.deps.renameProject).toHaveBeenCalledWith('a', 'New name');
    expect(order).toEqual(['rename', 'flush']);
  });

  it('routes deletion and closing at the project it was published for', async () => {
    const harness = createHarness();

    startBroker(harness.deps);
    harness.setOpenIds(['a']);

    const handle = getOpenProject('a')!;

    handle.markDeleted();
    await handle.deleteOnServer();
    handle.close();

    expect(harness.deps.markProjectDeleted).toHaveBeenCalledWith('a');
    expect(harness.deps.deleteProject).toHaveBeenCalledWith('a');
    expect(harness.deps.closeProject).toHaveBeenCalledWith('a');
  });

  /**
   * A rename is already in the reducer and the local snapshot by the time the flush runs, and the
   * next save retries the push. Failing the rename because the network blipped would undo nothing
   * and explain less — so the outcome is read by the surfaces that read the project back, not here.
   */
  it('does not fail a rename whose flush did not reach the server', async () => {
    const harness = createHarness();

    harness.deps.flushProject.mockResolvedValue({ documentJson: '{}', kind: 'unsynced' });
    startBroker(harness.deps);
    harness.setOpenIds(['a']);

    await expect(getOpenProject('a')!.rename('New name')).resolves.toBeUndefined();
    expect(harness.deps.renameProject).toHaveBeenCalledWith('a', 'New name');
  });

  /**
   * The registry is cleared when the account changes. A broker that trusted its own record of what
   * it had published would never register those projects again, and every library mutation would
   * silently take the HTTP path for the rest of the mount — the exact thing this exists to stop.
   */
  it('republishes its handles after the registry is cleared underneath it', async () => {
    const harness = createHarness();
    const account = await import('@platform/state/accountLifecycle');

    startBroker(harness.deps);
    harness.setOpenIds(['a']);
    expect(getOpenProject('a')).not.toBeNull();

    account.accountLifecycle.activate('someone-else');
    expect(getOpenProject('a')).toBeNull();

    harness.setOpenIds(['a', 'b']);
    expect(getOpenProject('a')).not.toBeNull();
    expect(getOpenProject('b')).not.toBeNull();
  });
});
