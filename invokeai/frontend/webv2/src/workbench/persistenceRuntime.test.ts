import type { HydratedWorkbenchSnapshot } from '@workbench/persistenceContracts';
import type { Project, WorkbenchState } from '@workbench/projectContracts';

import { describe, expect, it, vi } from 'vitest';

import type { ProjectRecoveredIdentity } from './projects/projectFlush';
import type { WorkbenchSaveResult } from './projects/syncedPersistence';

import {
  createWorkbenchPersistenceRuntime,
  type PersistenceAggregatePort,
  type PersistenceClock,
  type WorkbenchPersistencePort,
} from './persistenceRuntime';
import { createInitialWorkbenchState } from './workbenchState.testing';

const flushPromises = async (): Promise<void> => {
  await Promise.resolve();
  await Promise.resolve();
};

const deferred = <T>() => {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, reject, resolve };
};

const snapshot = (state: WorkbenchState, savedAt = '2026-07-17T00:00:00.000Z'): HydratedWorkbenchSnapshot => ({
  savedAt,
  state,
  version: 1,
});

const saveResult = (state: WorkbenchState, savedAt?: string): WorkbenchSaveResult => ({
  conflicts: [],
  deletedProjectForks: [],
  hasPendingChanges: false,
  projectBoardAssignments: [],
  snapshot: snapshot(state, savedAt),
});

class FakeClock implements PersistenceClock {
  private nextId = 0;
  private readonly callbacks = new Map<number, () => void>();

  clearTimeout(id: unknown): void {
    this.callbacks.delete(id as number);
  }

  runAll(): void {
    const callbacks = [...this.callbacks.values()];
    this.callbacks.clear();
    for (const callback of callbacks) {
      callback();
    }
  }

  setTimeout(callback: () => void): unknown {
    this.nextId += 1;
    this.callbacks.set(this.nextId, callback);
    return this.nextId;
  }
}

const recoveredIdentity = (projectId: string, name = 'Recovered'): ProjectRecoveredIdentity => ({
  id: 'recovered',
  name,
  recoveredAt: '2026-08-07T00:00:00.000Z',
  recoveryOf: projectId,
});

/**
 * What the real reducers do with a fork: keep the live project's content and re-label it with the
 * server's identity. Mirrored here rather than simplified to `recoveredProject`, because the runtime
 * applies these results even when the save that produced them is stale — and a fake that swapped in
 * the snapshot would agree with a runtime that silently discarded the newest edits.
 */
const forkFromLive = (
  state: WorkbenchState,
  projectId: string,
  resolution: { recoveredIdentity: ProjectRecoveredIdentity; recoveredProject: Project }
): Project => {
  const live = state.projects.find((project) => project.id === projectId);

  return live
    ? {
        ...live,
        id: resolution.recoveredIdentity.id,
        name: resolution.recoveredIdentity.name,
        recoveredAt: resolution.recoveredIdentity.recoveredAt,
        recoveryOf: resolution.recoveredIdentity.recoveryOf,
      }
    : resolution.recoveredProject;
};

const createAggregate = (initialState = createInitialWorkbenchState()) => {
  let state = structuredClone(initialState);
  let revision = 0;
  let hasHydrated = false;
  const listeners = new Set<() => void>();
  const events: string[] = [];
  const emit = () => {
    for (const listener of listeners) {
      listener();
    }
  };
  const boardAssignments: { boardId: string; projectId: string }[] = [];
  const port: PersistenceAggregatePort = {
    assignProjectBoard: (assignment) => {
      boardAssignments.push(assignment);
      events.push('assignProjectBoard');
    },
    getPersistedRevision: () => revision,
    getState: () => state,
    hydrate: (nextState) => {
      state = structuredClone(nextState);
      revision += 1;
      events.push('hydrate');
      emit();
    },
    notifyProjectNotFound: () => events.push('not-found'),
    reconcileConflict: (conflict) => {
      state = {
        ...state,
        projects: [
          ...state.projects.filter((project) => project.id !== conflict.projectId),
          conflict.serverProject,
          forkFromLive(state, conflict.projectId, conflict),
        ],
      };
      revision += 1;
      events.push('conflict');
      emit();
    },
    reconcileDeletedProject: (fork) => {
      state = {
        ...state,
        projects: state.projects.map((project) =>
          project.id === fork.projectId ? forkFromLive(state, fork.projectId, fork) : project
        ),
      };
      revision += 1;
      events.push('deleted-fork');
      emit();
    },
    reportLoadError: (error) => events.push(`load-error:${error}`),
    saveFailed: (error) => events.push(`save-failed:${error}`),
    saveStarted: () => events.push('save-started'),
    saveSucceeded: (savedAt) => events.push(`save-succeeded:${savedAt}`),
    setHasHydrated: (next) => {
      hasHydrated = next;
      events.push(`hydrated:${next}`);
      emit();
    },
    subscribe: (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };

  return {
    boardAssignments,
    connect() {
      state = { ...state, backendConnection: { status: 'connected' } };
      emit();
    },
    /** `patch` is for content a fork's identity does not overwrite, so a test can tell them apart. */
    edit(name = 'Edited', patch: Partial<Project> = {}) {
      state = {
        ...state,
        projects: state.projects.map((project, index) => (index === 0 ? { ...project, ...patch, name } : project)),
      };
      revision += 1;
      emit();
    },
    events,
    get hasHydrated() {
      return hasHydrated;
    },
    get revision() {
      return revision;
    },
    get state() {
      return state;
    },
    port,
  };
};

const createPersistence = (load: WorkbenchPersistencePort['loadWorkbench']) => {
  let pending = false;
  const persistence: WorkbenchPersistencePort = {
    hasPendingChanges: () => pending,
    loadWorkbench: vi.fn(load),
    saveWorkbench: vi.fn((state) => Promise.resolve(saveResult(state))),
  };
  return {
    persistence,
    setPending(next: boolean) {
      pending = next;
    },
  };
};

describe('Workbench persistence runtime', () => {
  it('hydrates before enabling saves and publishes lifecycle status', async () => {
    const aggregate = createAggregate();
    const loadedState = createInitialWorkbenchState();
    loadedState.projects[0]!.name = 'Loaded';
    const { persistence } = createPersistence(() => Promise.resolve(snapshot(loadedState)));
    const clock = new FakeClock();
    const runtime = createWorkbenchPersistenceRuntime({ aggregate: aggregate.port, clock, persistence });
    const phases: string[] = [];
    runtime.subscribe(() => phases.push(runtime.getSnapshot().phase));

    runtime.start();
    expect(runtime.getSnapshot().phase).toBe('hydrating');
    expect(persistence.saveWorkbench).not.toHaveBeenCalled();
    await flushPromises();

    expect(aggregate.state.projects[0]?.name).toBe('Loaded');
    expect(aggregate.hasHydrated).toBe(true);
    expect(phases).toEqual(['hydrating', 'idle']);
    clock.runAll();
    expect(persistence.saveWorkbench).not.toHaveBeenCalled();
  });

  it('reports load failure, hydrates the shell, and remains usable', async () => {
    const aggregate = createAggregate();
    const { persistence } = createPersistence(() => Promise.reject(new Error('server unavailable')));
    const runtime = createWorkbenchPersistenceRuntime({
      aggregate: aggregate.port,
      clock: new FakeClock(),
      persistence,
    });

    runtime.start();
    await flushPromises();

    expect(aggregate.events).toContain('load-error:server unavailable');
    expect(aggregate.hasHydrated).toBe(true);
    expect(runtime.getSnapshot()).toEqual({ error: null, phase: 'idle' });
  });

  it('preserves an edit made during load and saves it only after load settles', async () => {
    const aggregate = createAggregate();
    const load = deferred<HydratedWorkbenchSnapshot | null>();
    const { persistence } = createPersistence(() => load.promise);
    const clock = new FakeClock();
    const runtime = createWorkbenchPersistenceRuntime({ aggregate: aggregate.port, clock, persistence });

    runtime.start();
    aggregate.edit('Local edit during load');
    clock.runAll();
    expect(persistence.saveWorkbench).not.toHaveBeenCalled();

    const remote = createInitialWorkbenchState();
    remote.projects[0]!.name = 'Remote';
    load.resolve(snapshot(remote));
    await flushPromises();
    clock.runAll();

    expect(aggregate.state.projects[0]?.name).toBe('Local edit during load');
    expect(persistence.saveWorkbench).toHaveBeenCalledWith(
      expect.objectContaining({ projects: [expect.objectContaining({ name: 'Local edit during load' })] })
    );
  });

  it('debounces edits and ignores stale completions after a newer revision', async () => {
    const aggregate = createAggregate();
    const { persistence } = createPersistence(() => Promise.resolve(null));
    const first = deferred<WorkbenchSaveResult>();
    const second = deferred<WorkbenchSaveResult>();
    vi.mocked(persistence.saveWorkbench)
      .mockImplementationOnce(() => first.promise)
      .mockImplementationOnce(() => second.promise);
    const clock = new FakeClock();
    const runtime = createWorkbenchPersistenceRuntime({ aggregate: aggregate.port, clock, persistence });

    runtime.start();
    await flushPromises();
    aggregate.edit('First');
    aggregate.edit('Second');
    expect(persistence.saveWorkbench).not.toHaveBeenCalled();
    clock.runAll();
    expect(persistence.saveWorkbench).toHaveBeenCalledTimes(1);

    aggregate.edit('Third');
    clock.runAll();
    first.resolve(saveResult(aggregate.state, 'stale'));
    await flushPromises();
    expect(aggregate.events).not.toContain('save-succeeded:stale');

    second.resolve(saveResult(aggregate.state, 'current'));
    await flushPromises();
    expect(aggregate.events).toContain('save-succeeded:current');
  });

  it('applies server outcomes from a save that went stale', async () => {
    // A stale save is one whose *snapshot* moved on, which says nothing about what the server
    // answered. The board id in a create response exists nowhere else, and the first save of a new
    // draft is exactly the one an edit is most likely to overtake — so dropping it there leaves the
    // project pointing at no board until the next reload.
    const aggregate = createAggregate();
    const { persistence } = createPersistence(() => Promise.resolve(null));
    const first = deferred<WorkbenchSaveResult>();
    const second = deferred<WorkbenchSaveResult>();
    vi.mocked(persistence.saveWorkbench)
      .mockImplementationOnce(() => first.promise)
      .mockImplementationOnce(() => second.promise);
    const clock = new FakeClock();
    const runtime = createWorkbenchPersistenceRuntime({ aggregate: aggregate.port, clock, persistence });

    runtime.start();
    await flushPromises();
    aggregate.edit('First');
    clock.runAll();

    // The edit that makes the in-flight save stale.
    aggregate.edit('Second');
    clock.runAll();

    first.resolve({
      ...saveResult(aggregate.state, 'stale'),
      projectBoardAssignments: [{ boardId: 'board-1', projectId: 'project-1' }],
    });
    await flushPromises();

    expect(aggregate.boardAssignments).toEqual([{ boardId: 'board-1', projectId: 'project-1' }]);
    // Still stale for the purposes of the save's own bookkeeping.
    expect(aggregate.events).not.toContain('save-succeeded:stale');

    second.resolve(saveResult(aggregate.state, 'current'));
    await flushPromises();
    expect(aggregate.events).toContain('save-succeeded:current');
  });

  it('applies a stale save’s deletion fork without discarding what landed mid-flight', async () => {
    // The two halves of this have to hold together. Dropping the fork because the save went stale
    // re-creates the deleted project under its old id on the next push; applying a fork built from
    // the pushed snapshot throws away the edit that made the save stale in the first place. The
    // answer is that a fork carries an identity, not a document.
    const aggregate = createAggregate();
    const { persistence } = createPersistence(() => Promise.resolve(null));
    const pending = deferred<WorkbenchSaveResult>();
    vi.mocked(persistence.saveWorkbench).mockImplementationOnce(() => pending.promise);
    const clock = new FakeClock();
    const runtime = createWorkbenchPersistenceRuntime({ aggregate: aggregate.port, clock, persistence });

    runtime.start();
    await flushPromises();
    aggregate.edit('Before the push', { settings: { ...aggregate.state.projects[0]!.settings, useCpuNoise: false } });

    const original = aggregate.state.projects[0]!;

    clock.runAll();

    // Lands while the save is in flight, which is what makes the result stale.
    aggregate.edit('After the push started', { settings: { ...original.settings, useCpuNoise: true } });

    pending.resolve({
      ...saveResult(aggregate.state, 'stale'),
      deletedProjectForks: [
        {
          projectId: original.id,
          recoveredIdentity: recoveredIdentity(original.id, 'Recovered'),
          // What the push was carrying, one edit behind.
          recoveredProject: { ...original, id: 'recovered', name: 'Before the push' },
        },
      ],
    });
    await flushPromises();

    // Applied despite the staleness: dropping it would let the next push re-create the deleted id.
    expect(aggregate.events).toContain('deleted-fork');
    expect(aggregate.state.projects.map((project) => project.id)).not.toContain(original.id);

    const fork = aggregate.state.projects.find((project) => project.id === 'recovered');

    expect(fork?.recoveryOf).toBe(original.id);
    // ...and the edit that made it stale is in the fork rather than lost between the two copies.
    expect(fork?.settings.useCpuNoise).toBe(true);
  });

  it('ignores server outcomes once disposed', async () => {
    const aggregate = createAggregate();
    const { persistence } = createPersistence(() => Promise.resolve(null));
    const pending = deferred<WorkbenchSaveResult>();
    vi.mocked(persistence.saveWorkbench).mockImplementationOnce(() => pending.promise);
    const clock = new FakeClock();
    const runtime = createWorkbenchPersistenceRuntime({ aggregate: aggregate.port, clock, persistence });

    runtime.start();
    await flushPromises();
    aggregate.edit('First');
    clock.runAll();
    runtime.dispose();

    pending.resolve({
      ...saveResult(aggregate.state),
      projectBoardAssignments: [{ boardId: 'board-1', projectId: 'project-1' }],
    });
    await flushPromises();

    expect(aggregate.boardAssignments).toEqual([]);
  });

  it('keeps one save in flight and coalesces queued edits into the latest state', async () => {
    const aggregate = createAggregate();
    const { persistence } = createPersistence(() => Promise.resolve(null));
    const first = deferred<WorkbenchSaveResult>();
    const second = deferred<WorkbenchSaveResult>();
    vi.mocked(persistence.saveWorkbench)
      .mockImplementationOnce(() => first.promise)
      .mockImplementationOnce(() => second.promise);
    const clock = new FakeClock();
    const runtime = createWorkbenchPersistenceRuntime({ aggregate: aggregate.port, clock, persistence });

    runtime.start();
    await flushPromises();
    aggregate.edit('First');
    clock.runAll();
    expect(persistence.saveWorkbench).toHaveBeenCalledTimes(1);

    aggregate.edit('Second');
    clock.runAll();
    aggregate.edit('Latest');
    clock.runAll();
    expect(persistence.saveWorkbench).toHaveBeenCalledTimes(1);

    first.resolve(saveResult(aggregate.state, 'first'));
    await flushPromises();
    expect(persistence.saveWorkbench).toHaveBeenCalledTimes(2);
    expect(persistence.saveWorkbench).toHaveBeenLastCalledWith(
      expect.objectContaining({ projects: [expect.objectContaining({ name: 'Latest' })] })
    );

    second.resolve(saveResult(aggregate.state, 'latest'));
    await flushPromises();
    expect(aggregate.events).toContain('save-succeeded:latest');
  });

  it('holds a failed revision until a new edit and then retries', async () => {
    const aggregate = createAggregate();
    const { persistence } = createPersistence(() => Promise.resolve(null));
    vi.mocked(persistence.saveWorkbench).mockRejectedValueOnce(new Error('offline'));
    const clock = new FakeClock();
    const runtime = createWorkbenchPersistenceRuntime({ aggregate: aggregate.port, clock, persistence });

    runtime.start();
    await flushPromises();
    aggregate.edit();
    clock.runAll();
    await flushPromises();
    expect(aggregate.events).toContain('save-failed:offline');

    clock.runAll();
    expect(persistence.saveWorkbench).toHaveBeenCalledTimes(1);
    aggregate.edit('Retry revision');
    clock.runAll();
    await flushPromises();
    expect(persistence.saveWorkbench).toHaveBeenCalledTimes(2);
  });

  it('replays pending work immediately on reconnect and rejects a stale replay', async () => {
    const aggregate = createAggregate();
    const { persistence, setPending } = createPersistence(() => Promise.resolve(null));
    const replay = deferred<WorkbenchSaveResult>();
    vi.mocked(persistence.saveWorkbench).mockImplementationOnce(() => replay.promise);
    const clock = new FakeClock();
    const runtime = createWorkbenchPersistenceRuntime({ aggregate: aggregate.port, clock, persistence });

    runtime.start();
    await flushPromises();
    aggregate.edit('Offline edit');
    setPending(true);
    aggregate.connect();
    expect(persistence.saveWorkbench).toHaveBeenCalledTimes(1);

    aggregate.edit('Edit during replay');
    replay.resolve(saveResult(aggregate.state, 'stale-replay'));
    await flushPromises();
    expect(aggregate.events).not.toContain('save-succeeded:stale-replay');
    clock.runAll();
    expect(persistence.saveWorkbench).toHaveBeenCalledTimes(2);
  });

  it('applies conflict recovery and schedules the reconciled fork', async () => {
    const aggregate = createAggregate();
    const { persistence } = createPersistence(() => Promise.resolve(null));
    const clock = new FakeClock();
    const runtime = createWorkbenchPersistenceRuntime({ aggregate: aggregate.port, clock, persistence });

    runtime.start();
    await flushPromises();
    aggregate.edit();
    const original = aggregate.state.projects[0]!;
    vi.mocked(persistence.saveWorkbench).mockResolvedValueOnce({
      conflicts: [
        {
          projectId: original.id,
          recoveredIdentity: recoveredIdentity(original.id),
          recoveredProject: { ...original, id: 'recovered', name: 'Recovered' },
          serverProject: { ...original, name: 'Server' },
        },
      ],
      deletedProjectForks: [],
      hasPendingChanges: false,
      projectBoardAssignments: [],
      snapshot: snapshot(aggregate.state),
    });
    clock.runAll();
    await flushPromises();

    expect(aggregate.events).toContain('conflict');
    expect(aggregate.state.projects.map((project) => project.name)).toEqual(['Server', 'Recovered']);
    clock.runAll();
    expect(persistence.saveWorkbench).toHaveBeenCalledTimes(2);
  });

  it('cancels timers and ignores load/save completions after disposal', async () => {
    const aggregate = createAggregate();
    const load = deferred<HydratedWorkbenchSnapshot | null>();
    const { persistence } = createPersistence(() => load.promise);
    const clock = new FakeClock();
    const runtime = createWorkbenchPersistenceRuntime({ aggregate: aggregate.port, clock, persistence });
    const listener = vi.fn();
    runtime.subscribe(listener);

    runtime.start();
    runtime.dispose();
    load.resolve(snapshot(createInitialWorkbenchState()));
    await flushPromises();
    aggregate.edit();
    clock.runAll();

    expect(runtime.getSnapshot().phase).toBe('disposed');
    expect(aggregate.hasHydrated).toBe(false);
    expect(persistence.saveWorkbench).not.toHaveBeenCalled();
    expect(listener).toHaveBeenCalledTimes(1);
  });

  it('disposes immediately when its owning account signal is aborted', async () => {
    const aggregate = createAggregate();
    const { persistence } = createPersistence(() => Promise.resolve(null));
    const clock = new FakeClock();
    const controller = new AbortController();
    const runtime = createWorkbenchPersistenceRuntime({
      aggregate: aggregate.port,
      clock,
      persistence,
      signal: controller.signal,
    });

    runtime.start();
    await flushPromises();
    aggregate.edit('Account A edit');

    controller.abort();
    clock.runAll();

    expect(runtime.getSnapshot().phase).toBe('disposed');
    expect(persistence.saveWorkbench).not.toHaveBeenCalled();
  });

  it('hydrates through a fresh instance after a prior one was disposed mid-load (StrictMode remount)', async () => {
    const aggregate = createAggregate();
    const loadedState = createInitialWorkbenchState();
    loadedState.projects[0]!.name = 'Loaded';
    const { persistence } = createPersistence(() => Promise.resolve(snapshot(loadedState)));
    const clock = new FakeClock();

    const first = createWorkbenchPersistenceRuntime({ aggregate: aggregate.port, clock, persistence });
    first.start();
    first.dispose();

    const second = createWorkbenchPersistenceRuntime({ aggregate: aggregate.port, clock, persistence });
    second.start();
    await flushPromises();

    expect(aggregate.hasHydrated).toBe(true);
    expect(aggregate.state.projects[0]?.name).toBe('Loaded');
    expect(second.getSnapshot()).toEqual({ error: null, phase: 'idle' });
  });
});
