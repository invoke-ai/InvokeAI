import type { HydratedWorkbenchSnapshot } from '@workbench/persistenceContracts';
import type { Project, WorkbenchState } from '@workbench/projectContracts';

import { describe, expect, it, vi } from 'vitest';

import type { WorkbenchSaveResult } from './projects/syncedPersistence';

import {
  createWorkbenchPersistenceRuntime,
  type PersistenceAggregatePort,
  type PersistenceClock,
  type WorkbenchPersistencePort,
} from './persistenceRuntime';
import { WorkbenchBackendUnavailableError } from './projects/syncedPersistence';
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
  refusedProjects: [],
  savedAt,
  state,
  version: 1,
});

const saveResult = (state: WorkbenchState, savedAt?: string): WorkbenchSaveResult => ({
  conflicts: [],
  error: null,
  hasPendingChanges: false,
  localDraftStatus: 'ok',
  projectBoardAssignments: [],
  shouldRetry: false,
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
    reportLoadAvailable: () => events.push('load-available'),
    reportRefusedProjects: (refused) => events.push(`refused-projects:${refused.map((r) => r.projectId).join(',')}`),
    reportLoadError: (error) => events.push(`load-error:${error}`),
    reportLoadUnavailable: (error) => events.push(`load-unavailable:${error}`),
    saveFailed: (error) => events.push(`save-failed:${error}`),
    savePending: (error) => events.push(`save-pending:${error}`),
    saveScheduled: () => events.push('save-scheduled'),
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
    expect(aggregate.events.indexOf('hydrate')).toBeLessThan(aggregate.events.indexOf('load-available'));
    expect(phases).toEqual(['hydrating', 'idle']);
    clock.runAll();
    expect(persistence.saveWorkbench).not.toHaveBeenCalled();
  });

  it('blocks the editor on backend-unavailable load and retries without enabling saves', async () => {
    const aggregate = createAggregate();
    const loaded = snapshot(createInitialWorkbenchState());
    const { persistence } = createPersistence(
      vi
        .fn()
        .mockRejectedValueOnce(new WorkbenchBackendUnavailableError(new Error('server unavailable')))
        .mockResolvedValueOnce(loaded)
    );
    const runtime = createWorkbenchPersistenceRuntime({
      aggregate: aggregate.port,
      clock: new FakeClock(),
      persistence,
    });

    runtime.start();
    await flushPromises();

    expect(aggregate.events).toContain('load-unavailable:The project backend is unavailable.');
    expect(aggregate.hasHydrated).toBe(false);
    expect(runtime.getSnapshot()).toEqual({ error: 'The project backend is unavailable.', phase: 'unavailable' });
    expect(persistence.saveWorkbench).not.toHaveBeenCalled();

    runtime.retryLoad();
    await flushPromises();

    expect(aggregate.hasHydrated).toBe(true);
    expect(aggregate.events).toContain('load-available');
    expect(runtime.getSnapshot()).toEqual({ error: null, phase: 'idle' });
  });

  it('reports refused projects, leaving a deep-linked refusal to the session controller', async () => {
    const aggregate = createAggregate();
    const loaded = createInitialWorkbenchState();
    const refuse = (projectId: string) => ({
      projectId,
      projectName: projectId,
      raw: {},
      refusal: { raw: {}, scope: 'state' as const, status: 'unsupported-version' as const, version: 3 },
      source: 'canvas' as const,
    });
    const { persistence } = createPersistence(() =>
      Promise.resolve({ ...snapshot(loaded), refusedProjects: [refuse('future'), refuse('other')] })
    );
    const runtime = createWorkbenchPersistenceRuntime({
      aggregate: aggregate.port,
      clock: new FakeClock(),
      loadOptions: { openProjectId: 'future' },
      persistence,
    });

    runtime.start();
    await flushPromises();

    expect(aggregate.events).toContain('refused-projects:other');
    expect(aggregate.events).not.toContain('not-found');
    expect(aggregate.state.projects.map((project) => project.id)).toEqual(loaded.projects.map((project) => project.id));
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

  it('reports edits as pending from the moment they happen until their own save is acknowledged', async () => {
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
    aggregate.events.length = 0;

    // Before the debounce fires, the edit is already reported, so nothing can claim it was saved.
    aggregate.edit('First');
    await flushPromises();
    expect(aggregate.events).toEqual(['save-scheduled']);
    expect(persistence.saveWorkbench).not.toHaveBeenCalled();

    clock.runAll();
    expect(aggregate.events).toEqual(['save-scheduled', 'save-started']);

    // An edit made while the save runs makes that save's answer stale: nothing reports "saved" until the edit's
    // own save is acknowledged.
    aggregate.edit('Second');
    await flushPromises();
    first.resolve(saveResult(aggregate.state, 'first'));
    await flushPromises();
    expect(aggregate.events).toEqual(['save-scheduled', 'save-started']);

    clock.runAll();
    second.resolve(saveResult(aggregate.state, 'second'));
    await flushPromises();
    expect(aggregate.events).toEqual(['save-scheduled', 'save-started', 'save-started', 'save-succeeded:second']);
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
    // A stale snapshot can still receive the draft's only authoritative board id; preserve that create response
    // despite intervening edits.
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

  it('applies acknowledged board identities before reporting a partial save failure', async () => {
    const aggregate = createAggregate();
    const { persistence } = createPersistence(() => Promise.resolve(null));
    vi.mocked(persistence.saveWorkbench).mockResolvedValueOnce({
      ...saveResult(aggregate.state),
      error: 'Project B could not be saved.',
      hasPendingChanges: true,
      projectBoardAssignments: [{ boardId: 'board-a', projectId: 'project-a' }],
    });
    const clock = new FakeClock();
    const runtime = createWorkbenchPersistenceRuntime({ aggregate: aggregate.port, clock, persistence });

    runtime.start();
    await flushPromises();
    aggregate.edit();
    clock.runAll();
    await flushPromises();

    expect(aggregate.boardAssignments).toEqual([{ boardId: 'board-a', projectId: 'project-a' }]);
    expect(aggregate.events).toContain('save-failed:Project B could not be saved.');
    expect(runtime.getSnapshot()).toEqual({ error: 'Project B could not be saved.', phase: 'idle' });
  });

  it('retries transient work while reporting a hard failure from the same save', async () => {
    const aggregate = createAggregate();
    const { persistence } = createPersistence(() => Promise.resolve(null));
    vi.mocked(persistence.saveWorkbench)
      .mockResolvedValueOnce({
        ...saveResult(aggregate.state),
        error: 'Project A is too large.',
        hasPendingChanges: true,
        shouldRetry: true,
      })
      .mockResolvedValueOnce({
        ...saveResult(aggregate.state),
        error: 'Project A is too large.',
        hasPendingChanges: true,
        shouldRetry: false,
      });
    const clock = new FakeClock();
    const runtime = createWorkbenchPersistenceRuntime({ aggregate: aggregate.port, clock, persistence });

    runtime.start();
    await flushPromises();
    aggregate.edit();
    clock.runAll();
    await flushPromises();

    expect(aggregate.events).toContain('save-failed:Project A is too large.');
    expect(persistence.saveWorkbench).toHaveBeenCalledOnce();

    clock.runAll();
    await flushPromises();
    expect(persistence.saveWorkbench).toHaveBeenCalledTimes(2);
  });

  it('retries a transiently pending save without reporting it as saved', async () => {
    const aggregate = createAggregate();
    const { persistence } = createPersistence(() => Promise.resolve(null));
    vi.mocked(persistence.saveWorkbench)
      .mockResolvedValueOnce({
        ...saveResult(aggregate.state, 'pending'),
        hasPendingChanges: true,
        shouldRetry: true,
      })
      .mockResolvedValueOnce(saveResult(aggregate.state, 'acknowledged'));
    const clock = new FakeClock();
    const runtime = createWorkbenchPersistenceRuntime({ aggregate: aggregate.port, clock, persistence });
    runtime.start();
    await flushPromises();
    aggregate.edit();
    clock.runAll();
    await flushPromises();

    expect(aggregate.events).not.toContain('save-succeeded:pending');
    expect(aggregate.events).toContain('save-pending:Autosave is pending and will retry.');
    expect(persistence.saveWorkbench).toHaveBeenCalledOnce();

    clock.runAll();
    await flushPromises();
    expect(persistence.saveWorkbench).toHaveBeenCalledTimes(2);
    expect(aggregate.events).toContain('save-succeeded:acknowledged');
  });

  it('leaves a non-retrying conflict save in an explicit attention state', async () => {
    const aggregate = createAggregate();
    const { persistence } = createPersistence(() => Promise.resolve(null));
    vi.mocked(persistence.saveWorkbench).mockResolvedValueOnce({
      ...saveResult(aggregate.state),
      hasPendingChanges: true,
      shouldRetry: false,
    });
    const clock = new FakeClock();
    const runtime = createWorkbenchPersistenceRuntime({ aggregate: aggregate.port, clock, persistence });
    runtime.start();
    await flushPromises();
    aggregate.edit();
    clock.runAll();
    await flushPromises();

    expect(aggregate.events).toContain('save-pending:Autosave requires your attention.');
    expect(aggregate.events.some((event) => event.startsWith('save-succeeded:'))).toBe(false);
    clock.runAll();
    expect(persistence.saveWorkbench).toHaveBeenCalledOnce();
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
