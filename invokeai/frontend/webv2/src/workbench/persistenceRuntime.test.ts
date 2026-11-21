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
  awaitsDecision: false,
  conflicts: [],
  error: null,
  hasPendingChanges: false,
  hasRetryablePendingChanges: false,
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
    saveNeedsAttention: (savedAt) => events.push(`save-needs-attention:${savedAt}`),
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
  let unloadWouldLose = false;
  const sessionReopenedListeners = new Set<() => void>();
  const persistence: WorkbenchPersistencePort = {
    hasChangesUnloadWouldLose: () => unloadWouldLose,
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
    /** What local recovery reports after the latest journal: an unsaved document it could not take. */
    setUnloadWouldLose(next: boolean) {
      unloadWouldLose = next;
    },
  };
};

const createPage = () => {
  let onHidden: (() => void) | null = null;
  let shouldPrompt: (() => boolean) | null = null;
  const port: PageLifecyclePort = {
    subscribeBeforeUnload: (listener) => {
      shouldPrompt = listener;
      return () => {
        shouldPrompt = null;
      };
    },
    subscribeHidden: (listener) => {
      onHidden = listener;
      return () => {
        onHidden = null;
      };
    },
  };
  return {
    /** True when the browser would offer to stay on the page. */
    beforeUnload: () => shouldPrompt?.() ?? false,
    hide: () => onHidden?.(),
    /** Whether a `beforeunload` listener is registered at all. */
    get isListeningBeforeUnload() {
      return shouldPrompt !== null;
    },
    port,
  };
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

  it('journals held drafts before saving once across repeated unload and hidden checkpoints, including an in-flight save', async () => {
    const page = createPage();
    const draft = createHeldDraft();
    const { aggregate, clock, persistence } = await startLoaded({ commitDrafts: draft.commit, page: page.port });
    draft.held.commitTo = aggregate;
    const journaledNames = journalTo(persistence);

    page.hide();
    expect(journaledNames).toEqual([]);

    draft.held.name = 'Typed just before hiding';
    expect(page.beforeUnload()).toBe(false);
    // Before unload, pagehide and becoming hidden checkpoint the same revision.
    page.hide();
    page.hide();
    expect(journaledNames).toEqual(['Typed just before hiding']);
    expect(savedNames(persistence)).toEqual(['Typed just before hiding']);
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
});

describe('Workbench persistence runtime unload prompt', () => {
  /** A journal that records what it was given; a project named `tooLarge` is one local recovery cannot take. */
  const tooLarge = 'Too large to journal';
  const journalTo = (fake: ReturnType<typeof createPersistence>) => {
    const journaledNames: (string | undefined)[] = [];
    fake.persistence.journalBeforeUnload = vi.fn((state: WorkbenchState) => {
      const name = state.projects[0]?.name;
      journaledNames.push(name);
      fake.setUnloadWouldLose(name === tooLarge);
      return {
        kind: 'journaled' as const,
        projectIds: name === tooLarge ? [] : [state.projects[0]!.id],
        skippedOversizedProjectIds: name === tooLarge ? [state.projects[0]!.id] : [],
        written: Promise.resolve(true),
      };
    });
    return journaledNames;
  };

  it('checkpoints on every unload and asks when journaling fails outright', async () => {
    const page = createPage();
    const { aggregate, persistence } = await startLoaded({ page: page.port });
    expect(page.isListeningBeforeUnload).toBe(true);
    expect(page.beforeUnload()).toBe(false);
    persistence.journalBeforeUnload = vi.fn(() => {
      throw new Error('storage gone');
    });
    aggregate.edit('Unjournaled');
    expect(page.beforeUnload()).toBe(true);
    expect(savedNames(persistence)).toEqual(['Unjournaled']);
  });

  it('asks only when the journal could not take the unsaved revision, journaling it first', async () => {
    const page = createPage();
    const draft = createHeldDraft();
    const loaded = await startLoaded({ commitDrafts: draft.commit, page: page.port });
    const { aggregate, clock, persistence } = loaded;
    draft.held.commitTo = aggregate;
    const journaledNames = journalTo(loaded);
    aggregate.edit('At risk');
    clock.runAll();
    await flushPromises();
    expect(page.isListeningBeforeUnload).toBe(true);

    expect(page.beforeUnload()).toBe(false);
    expect(journaledNames).toEqual([]);

    // An ordinary edit, still held in a debounced field: the journal keeps it, so leaving needs no prompt.
    const save = deferred<WorkbenchSaveResult>();
    vi.mocked(persistence.saveWorkbench).mockImplementationOnce(() => save.promise);
    draft.held.name = 'Typed';
    expect(page.beforeUnload()).toBe(false);
    expect(journaledNames).toEqual(['Typed']);

    draft.held.name = tooLarge;
    expect(page.beforeUnload()).toBe(true);
    // Asked again (the user stayed, then left once more) without journaling the same revision twice.
    expect(page.beforeUnload()).toBe(true);
    // The unload that follows has nothing new to journal.
    page.hide();
    expect(journaledNames).toEqual(['Typed', tooLarge]);

    // Once the server has it, nothing is left to lose.
    save.resolve(saveResult(aggregate.state));
    await vi.waitFor(() => expect(savedNames(persistence)).toEqual(['At risk', 'Typed', tooLarge]));
    await flushPromises();
    expect(page.beforeUnload()).toBe(false);
  });
});
