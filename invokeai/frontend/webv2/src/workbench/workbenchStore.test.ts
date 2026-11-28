import { DEFAULT_LOGGING_CONFIG } from '@platform/logging/contracts';
import { configureLogging, getLogSnapshot, resetLogging } from '@platform/logging/logger';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { WorkbenchInternalStore, WorkbenchSnapshot } from './workbenchStore';

import { getProjectWidgetValues } from './widgetState';
import { createInitialWorkbenchState } from './workbenchState';
import { createWorkbenchStore } from './workbenchStore';

const overlays = vi.hoisted(() => ({ closeWidgetOverlays: vi.fn() }));
vi.mock('@platform/ui/widgetOverlayRegistry', () => overlays);

const watchSelector = <Selected>(
  store: WorkbenchInternalStore,
  selector: (snapshot: WorkbenchSnapshot) => Selected,
  isEqual: (left: Selected, right: Selected) => boolean = Object.is
) => {
  let current = selector(store.getSnapshot());
  let changeCount = 0;

  const unsubscribe = store.subscribe(() => {
    const next = selector(store.getSnapshot());

    if (!isEqual(current, next)) {
      current = next;
      changeCount += 1;
    }
  });

  return {
    get changeCount() {
      return changeCount;
    },
    get current() {
      return current;
    },
    unsubscribe,
  };
};

describe('createWorkbenchStore', () => {
  beforeEach(() => {
    resetLogging();
    configureLogging({ ...DEFAULT_LOGGING_CONFIG, level: 'trace' });
  });

  it('blocks persistence retargets only for queue work that has reached the backend', () => {
    const runningState = createInitialWorkbenchState();
    const source = runningState.projects[0]!;
    runningState.projects[0] = {
      ...source,
      queue: {
        items: [
          { backendItemIds: [11], id: 'run-1', snapshot: {}, status: 'running' } as (typeof source.queue.items)[number],
        ],
      },
    };
    const runningStore = createWorkbenchStore(runningState);
    const payload = {
      boardId: 'copy-board',
      name: `${source.name} (copy)`,
      project: { ...source, id: 'copy-id' },
      projectId: source.id,
      sourceName: source.name,
      targetProjectId: 'copy-id',
    };

    expect(runningStore.internal.persistence.retargetProject(payload)).toEqual({
      ok: false,
      reason: 'active-queue-runs',
    });

    for (const status of ['completed', 'failed', 'cancelled'] as const) {
      const terminalStore = createWorkbenchStore({
        ...runningState,
        projects: [
          {
            ...runningState.projects[0]!,
            queue: {
              items: [
                {
                  backendItemIds: [11],
                  id: 'run-1',
                  snapshot: {},
                  status,
                } as (typeof source.queue.items)[number],
              ],
            },
          },
        ],
      });
      expect(terminalStore.internal.persistence.retargetProject(payload)).toEqual({ ok: true });
    }

    const pendingState = createInitialWorkbenchState();
    pendingState.projects[0] = {
      ...pendingState.projects[0]!,
      queue: {
        items: [{ id: 'pending-1', snapshot: {}, status: 'pending' } as (typeof source.queue.items)[number]],
      },
    };
    const pendingStore = createWorkbenchStore(pendingState);
    const pendingSource = pendingState.projects[0]!;
    expect(
      pendingStore.internal.persistence.retargetProject({
        ...payload,
        project: { ...pendingSource, id: 'copy-id' },
        projectId: pendingSource.id,
        sourceName: pendingSource.name,
      })
    ).toEqual({ ok: true });
  });

  it('blocks persistence retargets while durable cancellation is pending', () => {
    const state = createInitialWorkbenchState();
    const source = state.projects[0]!;
    state.projects[0] = {
      ...source,
      queue: {
        items: [
          {
            backendItemIds: [11],
            cancellationPending: true,
            id: 'run-1',
            snapshot: {},
            status: 'cancelled',
          } as (typeof source.queue.items)[number],
        ],
      },
    };
    const store = createWorkbenchStore(state);

    expect(
      store.internal.persistence.retargetProject({
        boardId: 'copy-board',
        name: `${source.name} (copy)`,
        project: { ...source, id: 'copy-id' },
        projectId: source.id,
        sourceName: source.name,
        targetProjectId: 'copy-id',
      })
    ).toEqual({ ok: false, reason: 'active-queue-runs' });
  });

  it('coordinates layout preset activation across every command caller', async () => {
    const deferred = new Map<string, { promise: Promise<void>; resolve: () => void }>();
    const loadLayoutWidgets = (_layout: unknown, workspaceId: string): Promise<void> => {
      let resolve!: () => void;
      const promise = new Promise<void>((next) => {
        resolve = next;
      });

      deferred.set(workspaceId, { promise, resolve });
      return promise;
    };
    const store = createWorkbenchStore(createInitialWorkbenchState(), { loadLayoutWidgets });

    const stripActivation = store.commands.layout.activatePreset('compose');
    const hotkeyActivation = store.commands.layout.activatePreset('edit');

    deferred.get('edit')?.resolve();
    await hotkeyActivation;
    deferred.get('compose')?.resolve();
    await stripActivation;

    expect(store.getSnapshot().activeProject.workspaceId).toBe('edit');
    expect(store.getSnapshot().activeProject.invocation).toMatchObject({ destination: 'canvas', sourceId: 'canvas' });
  });

  it('discards a pending preset when project focus changes before loading finishes', async () => {
    let resolve!: () => void;
    const loadLayoutWidgets = (): Promise<void> =>
      new Promise<void>((next) => {
        resolve = next;
      });
    const store = createWorkbenchStore(createInitialWorkbenchState(), { loadLayoutWidgets });
    const firstProjectId = store.getSnapshot().activeProject.id;
    const secondProject = store.commands.projects.create();

    store.commands.projects.switchTo(firstProjectId);
    const activation = store.commands.layout.activatePreset('edit');
    store.commands.projects.switchTo(secondProject.id);
    store.commands.projects.switchTo(firstProjectId);
    resolve();
    await activation;

    expect(store.getSnapshot().activeProject.workspaceId).toBe('compose');
  });

  it('notifies subscribers once for reducer changes and not for no-op reducer results', () => {
    const store = createWorkbenchStore();
    const listener = vi.fn();

    store.subscribe(listener);

    const projectId = store.getSnapshot().activeProject.id;

    store.commands.projects.rename(projectId, 'Renamed');

    expect(listener).toHaveBeenCalledTimes(1);
    expect(store.getSnapshot().activeProject.name).toBe('Renamed');

    store.commands.projects.rename(projectId, '   ');

    expect(listener).toHaveBeenCalledTimes(1);
    expect(store.getSnapshot().activeProject.name).toBe('Renamed');
  });

  it('bumps persisted revision only for autosaved state changes', () => {
    const store = createWorkbenchStore();
    const initialPersistedRevision = store.getPersistedRevision();

    store.commands.queue.setConnectionStatus({ status: 'connected' });

    expect(store.getPersistedRevision()).toBe(initialPersistedRevision);

    store.commands.notifications.add({ kind: 'info', title: 'Global notice' });

    expect(store.getPersistedRevision()).toBe(initialPersistedRevision);

    store.commands.projects.rename(store.getSnapshot().activeProject.id, 'Persisted rename');

    expect(store.getPersistedRevision()).toBe(initialPersistedRevision + 1);
  });

  it('forwards payload, positional, and nullary command forms into reducer actions', () => {
    const store = createWorkbenchStore();
    const projectId = store.getSnapshot().activeProject.id;

    store.commands.notifications.add({ kind: 'info', title: 'Payload form' });
    expect(store.getSnapshot().notifications[0]?.title).toBe('Payload form');

    store.commands.gallery.setSearchTerm('positional form', projectId);
    expect(getProjectWidgetValues(store.getSnapshot().activeProject, 'gallery').searchTerm).toBe('positional form');

    store.commands.notifications.clear();
    expect(store.getSnapshot().notifications).toHaveLength(0);
  });

  it('records recordError actions into project diagnostics', () => {
    const store = createWorkbenchStore();
    const projectId = store.getSnapshot().activeProject.id;

    const error = new Error('socket closed');

    store.commands.notifications.reportError({
      area: 'queue-runtime',
      context: { error, itemId: 'item-1' },
      message: 'Queue failed',
      namespace: 'queue',
    });

    expect(getLogSnapshot().entries).toMatchObject([
      {
        context: { itemId: 'item-1' },
        error: { message: 'socket closed', name: 'Error' },
        level: 'error',
        message: 'Queue failed',
        name: 'queue.queue-runtime',
        namespace: 'queue',
        source: { area: 'queue-runtime', namespace: 'queue', projectId },
      },
    ]);
    expect(store.getSnapshot().notifications[0]).toMatchObject({
      message: 'Queue failed: socket closed',
      title: 'Error',
    });
  });

  it('records autosave outcomes and queue item transitions with the history owner as the only reporter', () => {
    const store = createWorkbenchStore();
    const projectId = store.getSnapshot().activeProject.id;

    store.internal.persistence.saveStarted();
    store.internal.persistence.savePending('Autosave is pending and will retry.');
    store.internal.persistence.saveNeedsAttention('2026-10-14T09:00:00.000Z');
    store.internal.persistence.saveFailed('Backend rejected the revision');
    store.commands.queue.setStatus({ error: 'Out of memory', projectId, queueItemId: 'missing', status: 'failed' });

    expect(getLogSnapshot().entries).toMatchObject([
      {
        context: { queueItemId: 'missing', reason: 'Out of memory', status: 'failed' },
        level: 'error',
        message: 'Queue item failed: Out of memory',
        name: 'queue.item-failed',
        source: { area: 'history', namespace: 'queue', projectId },
      },
      {
        context: { reason: 'Backend rejected the revision' },
        level: 'error',
        name: 'persistence.autosave-failed',
        source: { area: 'autosave', namespace: 'persistence', projectId },
      },
      { level: 'warn', name: 'persistence.autosave-needs-attention' },
      {
        context: { reason: 'Autosave is pending and will retry.' },
        level: 'warn',
        name: 'persistence.autosave-pending',
      },
      { level: 'debug', name: 'persistence.autosave-started' },
    ]);
  });

  it('lets narrow selectors observe only their own state changes', () => {
    const store = createWorkbenchStore();
    const backendStatusWatcher = watchSelector(store, (snapshot) => snapshot.backendConnection.status);
    const activeProjectIdWatcher = watchSelector(store, (snapshot) => snapshot.activeProject.id);
    const projectCountWatcher = watchSelector(store, (snapshot) => snapshot.projects.length);
    const batchCountWatcher = watchSelector(store, (snapshot) =>
      Number(getProjectWidgetValues(snapshot.activeProject, 'generate').batchCount ?? 1)
    );

    store.commands.queue.setConnectionStatus({ status: 'connected' });

    expect(backendStatusWatcher.current).toBe('connected');
    expect(backendStatusWatcher.changeCount).toBe(1);
    expect(activeProjectIdWatcher.changeCount).toBe(0);
    expect(projectCountWatcher.changeCount).toBe(0);
    expect(batchCountWatcher.changeCount).toBe(0);

    store.commands.generation.setBatchCount(3);

    expect(batchCountWatcher.current).toBe(3);
    expect(batchCountWatcher.changeCount).toBe(1);
    expect(backendStatusWatcher.changeCount).toBe(1);
    expect(activeProjectIdWatcher.changeCount).toBe(0);
    expect(projectCountWatcher.changeCount).toBe(0);

    store.commands.projects.create();

    expect(projectCountWatcher.current).toBe(2);
    expect(projectCountWatcher.changeCount).toBe(1);
    expect(activeProjectIdWatcher.current).toBe(store.getSnapshot().activeProject.id);
    expect(activeProjectIdWatcher.changeCount).toBe(1);
    expect(backendStatusWatcher.changeCount).toBe(1);
  });

  it('composes same-turn functional widget patches from the latest values', () => {
    const store = createWorkbenchStore();
    store.commands.widgets.patchValues('upscale', { weights: { first: 0.75, second: 0.5 }, steps: 30 });

    store.commands.widgets.patchValues('upscale', (current) => ({
      weights: { ...(current.weights as Record<string, number>), first: 0.8 },
    }));
    store.commands.widgets.patchValues('upscale', (current) => ({
      weights: { ...(current.weights as Record<string, number>), second: 0.55 },
    }));

    expect(getProjectWidgetValues(store.getSnapshot().activeProject, 'upscale')).toMatchObject({
      steps: 30,
      weights: { first: 0.8, second: 0.55 },
    });
  });

  it('reads and writes the fenced project after a switch and skips a closed project', () => {
    const store = createWorkbenchStore();
    const firstProjectId = store.getSnapshot().activeProject.id;
    store.commands.widgets.patchValues('video', { steps: 12 });
    const lateUpdate = () =>
      store.commands.widgets.patchValues('video', (current) => ({ steps: Number(current.steps) + 1 }), firstProjectId);

    store.commands.projects.create();
    store.commands.widgets.patchValues('video', { steps: 50 });
    lateUpdate();

    const firstProject = store.getSnapshot().projects.find((project) => project.id === firstProjectId)!;
    expect(getProjectWidgetValues(firstProject, 'video').steps).toBe(13);
    expect(getProjectWidgetValues(store.getSnapshot().activeProject, 'video').steps).toBe(50);

    store.commands.projects.close(firstProjectId);
    const update = vi.fn(() => ({ steps: 99 }));
    store.commands.widgets.patchValues('video', update, firstProjectId);
    expect(update).not.toHaveBeenCalled();
    expect(getProjectWidgetValues(store.getSnapshot().activeProject, 'video').steps).toBe(50);
  });

  it('enforces project lifecycle invariants through observable command results', () => {
    const store = createWorkbenchStore();
    const firstProjectId = store.getSnapshot().activeProject.id;

    expect(store.commands.projects.switchTo('missing')).toEqual({ ok: false, reason: 'project-not-found' });
    expect(store.getSnapshot().activeProject.id).toBe(firstProjectId);

    const secondProject = store.commands.projects.create();

    expect(store.getSnapshot().activeProject.id).toBe(secondProject.id);
    expect(store.commands.projects.switchTo(firstProjectId)).toEqual({ ok: true });
    expect(store.getSnapshot().activeProject.id).toBe(firstProjectId);
    expect(store.commands.projects.close(firstProjectId)).toEqual({ ok: true });
    expect(store.getSnapshot().activeProject.id).toBe(secondProject.id);
    expect(store.commands.projects.close(secondProject.id)).toEqual({ ok: false, reason: 'last-project' });
    expect(store.getSnapshot().notifications).toEqual([]);
  });

  it.each([
    { cancellationPending: undefined, status: 'pending' as const },
    { cancellationPending: undefined, status: 'running' as const },
    { cancellationPending: true, status: 'cancelled' as const },
  ])('keeps a project open while queue work is active ($status)', ({ cancellationPending, status }) => {
    const state = createInitialWorkbenchState();
    const project = state.projects[0]!;
    state.projects[0] = {
      ...project,
      queue: {
        items: [
          {
            cancellable: true,
            ...(cancellationPending === undefined ? {} : { cancellationPending }),
            id: 'active-run',
            snapshot: {} as (typeof project.queue.items)[number]['snapshot'],
            status,
          },
        ],
      },
    };
    const store = createWorkbenchStore(state);

    expect(store.commands.projects.close(project.id)).toEqual({ ok: false, reason: 'active-queue-runs' });
    store.commands.projects.create();

    expect(store.commands.projects.close(project.id)).toEqual({ ok: false, reason: 'active-queue-runs' });
    expect(store.getSnapshot().projects.some((candidate) => candidate.id === project.id)).toBe(true);
  });
});

it('owns recovery entries before rendering, preserves edits, and discards removed or previous-project entries', () => {
  const initial = createInitialWorkbenchState();
  const project = initial.projects[0]!;
  const id = 'recovery-fixture';
  project.widgetInstances[id] = { ...Object.values(project.widgetInstances)[0]!, id };
  const store = createWorkbenchStore(initial);
  const first = store.getSnapshot().activeProject;
  const recovery = store.widgetRecovery.get(id!)!;
  recovery.fail('actions', new Error('Old render'));
  store.commands.widgets.patchInstanceValues(id!, { retained: true });
  expect(store.widgetRecovery.get(id!)).toBe(recovery);
  const removed = {
    ...store.getState(),
    projects: store.getState().projects.map((p) =>
      p.id === first.id
        ? {
            ...p,
            widgetInstances: Object.fromEntries(Object.entries(p.widgetInstances).filter(([key]) => key !== id)),
          }
        : p
    ),
  };
  store.internal.persistence.hydrate(removed);
  expect(store.widgetRecovery.has(id!)).toBe(false);
  store.internal.persistence.hydrate({ ...removed, projects: [first] });
  expect(store.widgetRecovery.get(id!)).not.toBe(recovery);
  expect(store.widgetRecovery.get(id!)!.getSnapshot().errors).toEqual({});
  const replacement = store.widgetRecovery.get(id!);
  store.commands.projects.create();
  store.commands.projects.switchTo(first.id);
  expect(store.widgetRecovery.get(id!)).not.toBe(replacement);
  expect(store.widgetRecovery.size).toBe(Object.keys(store.getSnapshot().activeProject.widgetInstances).length);
});
