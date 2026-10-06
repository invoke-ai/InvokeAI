import type { HydratedWorkbenchSnapshot } from '@workbench/persistenceContracts';
import type { Project, WorkbenchState } from '@workbench/projectContracts';

import { getLogSnapshot } from '@platform/logging/logger';
import { describe, expect, it, vi } from 'vitest';

import type { WorkbenchSaveResult } from './projects/syncedPersistence';

import {
  createWorkbenchPersistenceRuntime,
  type PersistenceAggregatePort,
  type PageLifecyclePort,
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
  let closedSession = false;
  const sessionReopenedListeners = new Set<() => void>();
  const persistence: WorkbenchPersistencePort = {
    hasClosedSession: () => closedSession,
    hasPendingChanges: () => pending,
    journalBeforeUnload: vi.fn(() => ({ kind: 'nothing-to-journal' as const })),
    loadWorkbench: vi.fn(load),
    saveWorkbench: vi.fn((state) => Promise.resolve(saveResult(state))),
    subscribeSessionReopened: (listener) => {
      sessionReopenedListeners.add(listener);
      return () => sessionReopenedListeners.delete(listener);
    },
  };
  return {
    persistence,
    /** The close did not complete: the service reopened the session and told its listeners. */
    reopenSession() {
      closedSession = false;
      for (const listener of sessionReopenedListeners) {
        listener();
      }
    },
    setClosedSession(next: boolean) {
      closedSession = next;
    },
    setPending(next: boolean) {
      pending = next;
    },
  };
};

const createPage = () => {
  let onHidden: (() => void) | null = null;
  const port: PageLifecyclePort = {
    subscribeHidden: (listener) => {
      onHidden = listener;
      return () => {
        onHidden = null;
      };
    },
  };
  return { hide: () => onHidden?.(), port };
};

/** A draft an editor holds outside the aggregate until `commit` (the draft registry's flush) commits it. */
const createHeldDraft = () => {
  const held = { commitTo: null as ReturnType<typeof createAggregate> | null, name: null as string | null };
  return {
    commit: vi.fn(() => {
      if (held.name !== null) {
        held.commitTo?.edit(held.name);
        held.name = null;
      }
    }),
    held,
  };
};

const savedNames = (persistence: WorkbenchPersistencePort): (string | undefined)[] =>
  vi.mocked(persistence.saveWorkbench).mock.calls.map(([state]) => state.projects[0]?.name);

/** The log is process-wide: read only what a test recorded after taking its mark. */
const logMark = (): number => getLogSnapshot().entries[0]?.sequence ?? 0;
const loggedSince = (mark: number): string[] =>
  getLogSnapshot()
    .entries.filter((entry) => entry.sequence > mark)
    .map((entry) => entry.name);

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

  it('ignores server outcomes once its account lifetime has ended', async () => {
    const aggregate = createAggregate();
    const { persistence } = createPersistence(() => Promise.resolve(null));
    const pending = deferred<WorkbenchSaveResult>();
    vi.mocked(persistence.saveWorkbench).mockImplementationOnce(() => pending.promise);
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
    aggregate.edit('First');
    clock.runAll();
    controller.abort();

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

  it('ends a lifetime that never loaded without saving, ignoring the late load', async () => {
    const aggregate = createAggregate();
    const load = deferred<HydratedWorkbenchSnapshot | null>();
    const { persistence } = createPersistence(() => load.promise);
    const clock = new FakeClock();
    const runtime = createWorkbenchPersistenceRuntime({ aggregate: aggregate.port, clock, persistence });
    const listener = vi.fn();
    runtime.subscribe(listener);

    runtime.start();
    const exit = runtime.exit();
    load.resolve(snapshot(createInitialWorkbenchState()));
    await flushPromises();
    aggregate.edit();
    clock.runAll();

    await expect(exit.settled).resolves.toBe('nothing-to-save');
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

  it('hydrates through a fresh instance after a prior one exited mid-load (StrictMode remount)', async () => {
    const aggregate = createAggregate();
    const loadedState = createInitialWorkbenchState();
    loadedState.projects[0]!.name = 'Loaded';
    const { persistence } = createPersistence(() => Promise.resolve(snapshot(loadedState)));
    const clock = new FakeClock();

    const first = createWorkbenchPersistenceRuntime({ aggregate: aggregate.port, clock, persistence });
    first.start();
    const previousExit = first.exit();

    const second = createWorkbenchPersistenceRuntime({ aggregate: aggregate.port, clock, persistence, previousExit });
    second.start();
    await previousExit.settled;
    await flushPromises();

    expect(aggregate.hasHydrated).toBe(true);
    expect(aggregate.state.projects[0]?.name).toBe('Loaded');
    expect(second.getSnapshot()).toEqual({ error: null, phase: 'idle' });
  });
});

const startLoaded = async (options: Partial<Parameters<typeof createWorkbenchPersistenceRuntime>[0]> = {}) => {
  const aggregate = createAggregate();
  const fake = createPersistence(() => Promise.resolve(null));
  const clock = new FakeClock();
  const runtime = createWorkbenchPersistenceRuntime({
    aggregate: aggregate.port,
    clock,
    persistence: fake.persistence,
    ...options,
  });
  runtime.start();
  await flushPromises();
  return { aggregate, clock, runtime, ...fake };
};

describe('Workbench persistence runtime exit checkpoint', () => {
  it('saves the latest state when the editor exits inside the debounce', async () => {
    const { aggregate, persistence, runtime } = await startLoaded();
    aggregate.edit('Latest');

    const exit = runtime.exit();

    await expect(exit.settled).resolves.toBe('saved');
    expect(savedNames(persistence)).toEqual(['Latest']);
  });

  it('saves the newest state once, after the save already in flight settles', async () => {
    const { aggregate, clock, persistence, runtime } = await startLoaded();
    const inFlight = deferred<WorkbenchSaveResult>();
    vi.mocked(persistence.saveWorkbench).mockImplementationOnce(() => inFlight.promise);
    aggregate.edit('In flight');
    clock.runAll();
    aggregate.edit('Queued');
    clock.runAll();
    aggregate.edit('Latest');

    const exit = runtime.exit();
    await flushPromises();
    expect(savedNames(persistence)).toEqual(['In flight']);

    inFlight.resolve(saveResult(aggregate.state));
    await expect(exit.settled).resolves.toBe('saved');
    clock.runAll();
    await flushPromises();

    expect(savedNames(persistence)).toEqual(['In flight', 'Latest']);
  });

  it('writes nothing more when the save in flight already covers the latest state', async () => {
    const { aggregate, clock, persistence, runtime } = await startLoaded();
    const inFlight = deferred<WorkbenchSaveResult>();
    vi.mocked(persistence.saveWorkbench).mockImplementationOnce(() => inFlight.promise);
    aggregate.edit('Latest');
    clock.runAll();

    const exit = runtime.exit();
    inFlight.resolve(saveResult(aggregate.state));

    await expect(exit.settled).resolves.toBe('nothing-to-save');
    expect(savedNames(persistence)).toEqual(['Latest']);
  });

  it('saves again at exit when the latest revision previously failed', async () => {
    const { aggregate, clock, persistence, runtime } = await startLoaded();
    vi.mocked(persistence.saveWorkbench).mockRejectedValueOnce(new Error('offline'));
    aggregate.edit('Failed once');
    clock.runAll();
    await flushPromises();

    await expect(runtime.exit().settled).resolves.toBe('saved');
    expect(savedNames(persistence)).toEqual(['Failed once', 'Failed once']);
  });

  it('writes nothing once the account lifetime ends, even mid-checkpoint', async () => {
    const controller = new AbortController();
    const { aggregate, clock, persistence, runtime } = await startLoaded({ signal: controller.signal });
    const inFlight = deferred<WorkbenchSaveResult>();
    vi.mocked(persistence.saveWorkbench).mockImplementationOnce(() => inFlight.promise);
    aggregate.edit('Account A');
    clock.runAll();
    aggregate.edit('Account A, later');

    const exit = runtime.exit();
    controller.abort();
    inFlight.resolve(saveResult(aggregate.state));

    await expect(exit.settled).resolves.toBe('abandoned');
    expect(savedNames(persistence)).toEqual(['Account A']);
  });

  it('writes nothing for an exit that begins after the account lifetime ended', async () => {
    const controller = new AbortController();
    const { aggregate, persistence, runtime } = await startLoaded({ signal: controller.signal });
    aggregate.edit('Account A');
    controller.abort();

    await expect(runtime.exit().settled).resolves.toBe('abandoned');
    expect(persistence.saveWorkbench).not.toHaveBeenCalled();
  });

  it('records a failed exit save without rejecting, retrying, or touching the aggregate', async () => {
    const { aggregate, clock, persistence, runtime } = await startLoaded();
    vi.mocked(persistence.saveWorkbench).mockRejectedValueOnce(new Error('offline'));
    aggregate.edit('Unsaved');
    const eventsBeforeExit = [...aggregate.events];
    const mark = logMark();

    await expect(runtime.exit().settled).resolves.toBe('failed');
    clock.runAll();
    await flushPromises();

    expect(loggedSince(mark)).toEqual(['persistence.exit-checkpoint-failed']);
    expect(persistence.saveWorkbench).toHaveBeenCalledOnce();
    expect(aggregate.events).toEqual(eventsBeforeExit);
  });

  it('saves committed state without waiting for the capture, then once more for what the capture brought in', async () => {
    const { aggregate, persistence, runtime } = await startLoaded();
    const pixels = deferred<void>();
    aggregate.edit('Committed');

    const exit = runtime.exit({
      beforeCapture: async () => {
        await pixels.promise;
        aggregate.edit('With pixels');
      },
    });
    await flushPromises();
    expect(savedNames(persistence)).toEqual(['Committed']);

    pixels.resolve();
    await expect(exit.settled).resolves.toBe('saved');
    expect(savedNames(persistence)).toEqual(['Committed', 'With pixels']);
  });

  it.each([
    ['rejects', () => Promise.reject(new Error('upload failed'))],
    [
      'throws synchronously',
      () => {
        throw new Error('no engine');
      },
    ],
  ])('still saves, once, and records it when the capture %s', async (_, beforeCapture) => {
    const { aggregate, persistence, runtime } = await startLoaded();
    aggregate.edit('Committed');
    const mark = logMark();

    await expect(runtime.exit({ beforeCapture }).settled).resolves.toBe('saved');

    expect(savedNames(persistence)).toEqual(['Committed']);
    expect(loggedSince(mark)).toEqual(['persistence.exit-capture-incomplete']);
  });

  it('stops waiting for a capture that never settles and records it', async () => {
    const { aggregate, clock, persistence, runtime } = await startLoaded();
    aggregate.edit('Committed');
    const mark = logMark();

    const exit = runtime.exit({ beforeCapture: () => new Promise<void>(() => {}) });
    await flushPromises();
    clock.runAll();

    await expect(exit.settled).resolves.toBe('saved');
    expect(savedNames(persistence)).toEqual(['Committed']);
    expect(loggedSince(mark)).toEqual(['persistence.exit-capture-incomplete']);
  });

  it('saves again at exit when the save in flight comes back with an error', async () => {
    const { aggregate, clock, persistence, runtime } = await startLoaded();
    const inFlight = deferred<WorkbenchSaveResult>();
    vi.mocked(persistence.saveWorkbench).mockImplementationOnce(() => inFlight.promise);
    aggregate.edit('Latest');
    clock.runAll();

    const exit = runtime.exit();
    inFlight.resolve({ ...saveResult(aggregate.state), error: 'server error' });

    await expect(exit.settled).resolves.toBe('saved');
    expect(savedNames(persistence)).toEqual(['Latest', 'Latest']);
  });

  it('reports an exit save the backend did not acknowledge as failed and records it', async () => {
    const { aggregate, persistence, runtime } = await startLoaded();
    vi.mocked(persistence.saveWorkbench).mockResolvedValueOnce({
      ...saveResult(aggregate.state),
      shouldRetry: true,
    });
    aggregate.edit('Unacknowledged');
    const mark = logMark();

    await expect(runtime.exit().settled).resolves.toBe('failed');

    expect(savedNames(persistence)).toEqual(['Unacknowledged']);
    expect(loggedSince(mark)).toEqual(['persistence.exit-checkpoint-incomplete']);
  });

  it('commits drafts editors still hold and saves them in the one checkpoint write', async () => {
    const draft = createHeldDraft();
    const { aggregate, clock, persistence, runtime } = await startLoaded({ commitDrafts: draft.commit });
    draft.held.commitTo = aggregate;
    draft.held.name = 'Drafted';

    await expect(runtime.exit().settled).resolves.toBe('saved');
    clock.runAll();
    await flushPromises();

    expect(savedNames(persistence)).toEqual(['Drafted']);
  });

  it('stops observing the aggregate and the page at exit', async () => {
    const page = createPage();
    const { aggregate, clock, persistence, runtime } = await startLoaded({ page: page.port });

    await expect(runtime.exit().settled).resolves.toBe('nothing-to-save');
    aggregate.edit('After exit');
    page.hide();
    clock.runAll();
    await flushPromises();

    expect(persistence.saveWorkbench).not.toHaveBeenCalled();
  });

  it('does not reopen projects after the editor closed its session to leave', async () => {
    const { aggregate, persistence, runtime, setClosedSession } = await startLoaded();
    aggregate.edit('Last tab');
    setClosedSession(true);

    await expect(runtime.exit().settled).resolves.toBe('nothing-to-save');
    expect(persistence.saveWorkbench).not.toHaveBeenCalled();
  });

  it('saves an edit committed while the session was being emptied once the session reopens', async () => {
    const { aggregate, clock, persistence, reopenSession, setClosedSession } = await startLoaded();
    setClosedSession(true);
    aggregate.edit('Typed during the close');
    await flushPromises();
    clock.runAll();
    expect(persistence.saveWorkbench).not.toHaveBeenCalled();
    expect(aggregate.events.at(-1)).toBe('save-scheduled');

    reopenSession();
    await flushPromises();

    expect(savedNames(persistence)).toEqual(['Typed during the close']);
    expect(aggregate.events.at(-1)).toBe('save-succeeded:2026-07-17T00:00:00.000Z');
    clock.runAll();
    expect(persistence.saveWorkbench).toHaveBeenCalledOnce();
  });

  it('does not save on a reopened session when nothing was requested while it was closed', async () => {
    const { aggregate, clock, persistence, reopenSession, setClosedSession } = await startLoaded();
    aggregate.edit('Before the close');
    clock.runAll();
    await flushPromises();
    setClosedSession(true);

    reopenSession();
    await flushPromises();
    clock.runAll();

    expect(savedNames(persistence)).toEqual(['Before the close']);
  });

  it('loads a remounted editor only after the previous exit checkpoint settles', async () => {
    const first = await startLoaded();
    const inFlight = deferred<WorkbenchSaveResult>();
    vi.mocked(first.persistence.saveWorkbench).mockImplementationOnce(() => inFlight.promise);
    first.aggregate.edit('In flight');
    first.clock.runAll();
    first.aggregate.edit('Latest');
    const previousExit = first.runtime.exit();

    const second = createPersistence(() => Promise.resolve(null));
    const remounted = createWorkbenchPersistenceRuntime({
      aggregate: createAggregate().port,
      clock: new FakeClock(),
      persistence: second.persistence,
      previousExit,
    });
    remounted.start();
    await new Promise((resolve) => {
      setTimeout(resolve, 0);
    });

    expect(remounted.getSnapshot().phase).toBe('hydrating');
    expect(second.persistence.loadWorkbench).not.toHaveBeenCalled();

    inFlight.resolve(saveResult(first.aggregate.state));
    await vi.waitFor(() => expect(second.persistence.loadWorkbench).toHaveBeenCalledOnce());

    expect(savedNames(first.persistence)).toEqual(['In flight', 'Latest']);
    expect(vi.mocked(second.persistence.loadWorkbench).mock.invocationCallOrder[0]).toBeGreaterThan(
      vi.mocked(first.persistence.saveWorkbench).mock.invocationCallOrder[1]!
    );
  });

  it('keeps a later editor waiting when the one in between left before it loaded', async () => {
    const first = await startLoaded();
    const inFlight = deferred<WorkbenchSaveResult>();
    vi.mocked(first.persistence.saveWorkbench).mockImplementationOnce(() => inFlight.promise);
    first.aggregate.edit('In flight');
    first.clock.runAll();
    const firstExit = first.runtime.exit();

    const between = createWorkbenchPersistenceRuntime({
      aggregate: createAggregate().port,
      clock: new FakeClock(),
      persistence: createPersistence(() => Promise.resolve(null)).persistence,
      previousExit: firstExit,
    });
    between.start();
    const betweenExit = between.exit();

    const last = createPersistence(() => Promise.resolve(null));
    createWorkbenchPersistenceRuntime({
      aggregate: createAggregate().port,
      clock: new FakeClock(),
      persistence: last.persistence,
      previousExit: betweenExit,
    }).start();
    await new Promise((resolve) => {
      setTimeout(resolve, 0);
    });
    expect(last.persistence.loadWorkbench).not.toHaveBeenCalled();

    inFlight.resolve(saveResult(first.aggregate.state));
    await expect(betweenExit.settled).resolves.toBe('nothing-to-save');
    await vi.waitFor(() => expect(last.persistence.loadWorkbench).toHaveBeenCalledOnce());
  });

  it('stops waiting at the bound and keeps the superseded checkpoint from writing', async () => {
    const first = await startLoaded();
    const hung = deferred<WorkbenchSaveResult>();
    vi.mocked(first.persistence.saveWorkbench).mockImplementationOnce(() => hung.promise);
    first.aggregate.edit('Hung');
    first.clock.runAll();
    first.aggregate.edit('Latest');
    const previousExit = first.runtime.exit();
    const mark = logMark();

    const second = createPersistence(() => Promise.resolve(null));
    const clock = new FakeClock();
    const remounted = createWorkbenchPersistenceRuntime({
      aggregate: createAggregate().port,
      clock,
      persistence: second.persistence,
      previousExit,
    });
    remounted.start();
    await flushPromises();
    expect(second.persistence.loadWorkbench).not.toHaveBeenCalled();

    clock.runAll();
    await flushPromises();
    expect(second.persistence.loadWorkbench).toHaveBeenCalledOnce();

    hung.resolve(saveResult(first.aggregate.state));
    await expect(previousExit.settled).resolves.toBe('abandoned');
    expect(savedNames(first.persistence)).toEqual(['Hung']);
    expect(loggedSince(mark)).toEqual(['persistence.exit-checkpoint-abandoned']);
  });
});

describe('Workbench persistence runtime page lifecycle', () => {
  it('saves the unsaved revision at once when the page is hidden', async () => {
    const page = createPage();
    const { aggregate, persistence } = await startLoaded({ page: page.port });

    page.hide();
    expect(persistence.saveWorkbench).not.toHaveBeenCalled();

    aggregate.edit('Before hiding');
    page.hide();
    expect(savedNames(persistence)).toEqual(['Before hiding']);
  });

  it('does not re-push a failed revision or one already in flight when the page is hidden', async () => {
    const page = createPage();
    const { aggregate, clock, persistence } = await startLoaded({ page: page.port });
    vi.mocked(persistence.saveWorkbench).mockRejectedValueOnce(new Error('offline'));
    aggregate.edit('Failed');
    clock.runAll();
    await flushPromises();

    page.hide();
    expect(savedNames(persistence)).toEqual(['Failed']);

    const inFlight = deferred<WorkbenchSaveResult>();
    vi.mocked(persistence.saveWorkbench).mockImplementationOnce(() => inFlight.promise);
    aggregate.edit('In flight');
    clock.runAll();
    page.hide();
    inFlight.resolve(saveResult(aggregate.state));
    await flushPromises();

    expect(savedNames(persistence)).toEqual(['Failed', 'In flight']);
  });

  it('commits held drafts before deciding whether the hidden page has anything to save', async () => {
    const page = createPage();
    const draft = createHeldDraft();
    const { aggregate, clock, persistence } = await startLoaded({ commitDrafts: draft.commit, page: page.port });
    draft.held.commitTo = aggregate;

    page.hide();
    expect(draft.commit).toHaveBeenCalledOnce();
    expect(persistence.saveWorkbench).not.toHaveBeenCalled();

    draft.held.name = 'Typed just before hiding';
    page.hide();
    expect(savedNames(persistence)).toEqual(['Typed just before hiding']);

    // The debounced save the commit scheduled is replaced, not repeated.
    clock.runAll();
    await flushPromises();
    expect(savedNames(persistence)).toEqual(['Typed just before hiding']);
  });

  it('writes nothing and leaves drafts alone while the editor is leaving with a closed session', async () => {
    const page = createPage();
    const draft = createHeldDraft();
    const { aggregate, clock, persistence, setClosedSession } = await startLoaded({
      commitDrafts: draft.commit,
      page: page.port,
    });
    draft.held.commitTo = aggregate;
    setClosedSession(true);

    aggregate.edit('After the empty session');
    clock.runAll();
    draft.held.name = 'Drafted after the empty session';
    page.hide();

    expect(draft.commit).not.toHaveBeenCalled();
    expect(persistence.saveWorkbench).not.toHaveBeenCalled();
  });

  it('leaves drafts alone when the page is hidden before loading or after its account lifetime ended', async () => {
    const page = createPage();
    const draft = createHeldDraft();
    const aggregate = createAggregate();
    draft.held.commitTo = aggregate;
    draft.held.name = 'Drafted';
    const load = deferred<HydratedWorkbenchSnapshot | null>();
    const { persistence } = createPersistence(() => load.promise);
    const controller = new AbortController();
    const runtime = createWorkbenchPersistenceRuntime({
      aggregate: aggregate.port,
      clock: new FakeClock(),
      commitDrafts: draft.commit,
      page: page.port,
      persistence,
      signal: controller.signal,
    });

    runtime.start();
    page.hide();
    expect(draft.commit).not.toHaveBeenCalled();

    load.resolve(null);
    await flushPromises();
    controller.abort();
    page.hide();

    expect(draft.commit).not.toHaveBeenCalled();
    expect(persistence.saveWorkbench).not.toHaveBeenCalled();
  });

  it('queues a hidden-page save behind the one in flight', async () => {
    const page = createPage();
    const { aggregate, clock, persistence } = await startLoaded({ page: page.port });
    const inFlight = deferred<WorkbenchSaveResult>();
    vi.mocked(persistence.saveWorkbench).mockImplementationOnce(() => inFlight.promise);
    aggregate.edit('In flight');
    clock.runAll();
    aggregate.edit('Hidden');

    page.hide();
    expect(savedNames(persistence)).toEqual(['In flight']);

    inFlight.resolve(saveResult(aggregate.state));
    await flushPromises();
    expect(savedNames(persistence)).toEqual(['In flight', 'Hidden']);
  });
});

describe('Workbench persistence runtime unload journal', () => {
  const journalTo = (
    persistence: WorkbenchPersistencePort,
    written: () => Promise<boolean> = () => Promise.resolve(true)
  ) => {
    const journaledNames: (string | undefined)[] = [];
    persistence.journalBeforeUnload = vi.fn((state: WorkbenchState) => {
      journaledNames.push(state.projects[0]?.name);
      return { kind: 'journaled' as const, projectIds: [], skippedOversizedProjectIds: [], written: written() };
    });
    return journaledNames;
  };

  it('journals committed drafts before the hidden-page save, including a revision whose save is in flight', async () => {
    const page = createPage();
    const draft = createHeldDraft();
    const { aggregate, clock, persistence } = await startLoaded({ commitDrafts: draft.commit, page: page.port });
    draft.held.commitTo = aggregate;
    const journaledNames = journalTo(persistence);

    page.hide();
    expect(journaledNames).toEqual([]);

    draft.held.name = 'Typed just before hiding';
    page.hide();
    // Unloading fires both `pagehide` and becoming hidden; the second has nothing new to journal.
    page.hide();
    expect(journaledNames).toEqual(['Typed just before hiding']);
    expect(vi.mocked(persistence.journalBeforeUnload!).mock.invocationCallOrder[0]).toBeLessThan(
      vi.mocked(persistence.saveWorkbench).mock.invocationCallOrder[0]!
    );
    await flushPromises();

    const inFlight = deferred<WorkbenchSaveResult>();
    vi.mocked(persistence.saveWorkbench).mockImplementationOnce(() => inFlight.promise);
    aggregate.edit('Saving');
    clock.runAll();
    // The page may unload before this save stages anything, so the journal covers it too.
    page.hide();
    expect(journaledNames).toEqual(['Typed just before hiding', 'Saving']);

    inFlight.resolve(saveResult(aggregate.state));
    await flushPromises();
    expect(savedNames(persistence)).toEqual(['Typed just before hiding', 'Saving']);
  });

  it('journals while the exit checkpoint saves, until it settles', async () => {
    const page = createPage();
    const { aggregate, persistence, runtime } = await startLoaded({ page: page.port });
    const journaledNames = journalTo(persistence);
    const checkpointSave = deferred<WorkbenchSaveResult>();
    vi.mocked(persistence.saveWorkbench).mockImplementationOnce(() => checkpointSave.promise);
    aggregate.edit('Left the editor');

    const exit = runtime.exit();
    // Leaving the editor and reloading at once: the checkpoint's save cannot finish.
    page.hide();
    expect(journaledNames).toEqual(['Left the editor']);

    checkpointSave.resolve(saveResult(aggregate.state));
    await expect(exit.settled).resolves.toBe('saved');
    page.hide();
    expect(journaledNames).toEqual(['Left the editor']);
  });

  it('does not journal for an exit a newer editor superseded', async () => {
    const page = createPage();
    const { aggregate, persistence, runtime } = await startLoaded({ page: page.port });
    const journaledNames = journalTo(persistence);
    vi.mocked(persistence.saveWorkbench).mockImplementationOnce(() => deferred<WorkbenchSaveResult>().promise);
    aggregate.edit('Superseded');

    runtime.exit().supersede();
    page.hide();

    expect(journaledNames).toEqual([]);
  });

  it('reports a journal that could not be written and still saves', async () => {
    const page = createPage();
    const { aggregate, persistence } = await startLoaded({ page: page.port });
    persistence.journalBeforeUnload = vi
      .fn<NonNullable<WorkbenchPersistencePort['journalBeforeUnload']>>()
      .mockReturnValueOnce({ kind: 'unavailable' })
      .mockImplementationOnce(() => {
        throw new Error('storage gone');
      })
      .mockReturnValueOnce({
        kind: 'journaled',
        projectIds: [],
        skippedOversizedProjectIds: ['huge'],
        written: Promise.resolve(true),
      });
    const mark = logMark();

    aggregate.edit('First');
    page.hide();
    aggregate.edit('Second');
    page.hide();
    aggregate.edit('Third');
    page.hide();

    // Newest first.
    expect(loggedSince(mark).filter((name) => name.startsWith('persistence.unload-journal'))).toEqual([
      'persistence.unload-journal-oversized',
      'persistence.unload-journal-unavailable',
      'persistence.unload-journal-unavailable',
    ]);
    expect(savedNames(persistence)).toEqual(['First']);
    // Saving goes on: the hidden-page saves queued behind the first still run.
    await flushPromises();
    await flushPromises();
    expect(savedNames(persistence)).toEqual(['First', 'Third']);
  });

  it('journals the revision again after a journal write that aborted', async () => {
    const page = createPage();
    const { aggregate, persistence } = await startLoaded({ page: page.port });
    let isWritten = false;
    const journaledNames = journalTo(persistence, () => Promise.resolve(isWritten));
    vi.mocked(persistence.saveWorkbench).mockImplementation(() => deferred<WorkbenchSaveResult>().promise);
    const mark = logMark();
    aggregate.edit('Hidden');

    page.hide();
    await flushPromises();
    isWritten = true;
    page.hide();

    expect(journaledNames).toEqual(['Hidden', 'Hidden']);
    expect(loggedSince(mark)).toContain('persistence.unload-journal-aborted');
  });

  it('does not journal, nor count as journaled, while the session is closed', async () => {
    const page = createPage();
    const { aggregate, persistence, runtime, setClosedSession } = await startLoaded({ page: page.port });
    const journaledNames = journalTo(persistence);
    vi.mocked(persistence.saveWorkbench).mockImplementation(() => deferred<WorkbenchSaveResult>().promise);
    aggregate.edit('Closing');
    runtime.exit();
    setClosedSession(true);
    page.hide();
    expect(journaledNames).toEqual([]);

    // The close did not complete and the session reopened: the same revision is journaled after all.
    setClosedSession(false);
    page.hide();
    expect(journaledNames).toEqual(['Closing']);
  });
});
