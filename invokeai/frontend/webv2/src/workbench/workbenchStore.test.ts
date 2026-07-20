import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { WorkbenchInternalStore, WorkbenchSnapshot } from './workbenchStore';

import { clearProjectDiagnostics, configureDiagnostics, getProjectDiagnostics } from './diagnostics/logger';
import { areWidgetPlacementProjectsEqual, getWidgetPlacementProject } from './widgetPlacementMeta';
import { getProjectWidgetValues } from './widgetState';
import { createInitialWorkbenchState } from './workbenchState';
import { createWorkbenchStore } from './workbenchStore';

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
    configureDiagnostics({
      enabled: true,
      level: 'trace',
      namespaces: ['system', 'queue'],
      performanceTimingsEnabled: false,
    });
  });

  it('exposes stable capability interfaces and initializes snapshot metadata', () => {
    const store = createWorkbenchStore();
    const snapshot = store.getSnapshot();

    expect(store.commands).toBe(store.commands);
    expect(store.queries).toBe(store.queries);
    expect(snapshot.hasHydrated).toBe(false);
    expect(snapshot.projects).toHaveLength(1);
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

  it('updates hydration metadata without changing durable workbench state', () => {
    const store = createWorkbenchStore();
    const initialState = store.getState();
    const initialPersistedRevision = store.getPersistedRevision();
    const listener = vi.fn();

    store.subscribe(listener);

    store.setHasHydrated(true);

    expect(listener).toHaveBeenCalledTimes(1);
    expect(store.getSnapshot().hasHydrated).toBe(true);
    expect(store.getState()).toBe(initialState);
    expect(store.getPersistedRevision()).toBe(initialPersistedRevision);

    store.setHasHydrated(true);

    expect(listener).toHaveBeenCalledTimes(1);
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

  it('forwards payload-form commands into reducer actions', () => {
    const store = createWorkbenchStore();

    store.commands.notifications.add({ kind: 'info', title: 'Payload form' });

    expect(store.getSnapshot().notifications[0]?.title).toBe('Payload form');
  });

  it('maps positional-form commands onto their action payloads', () => {
    const store = createWorkbenchStore();
    const projectId = store.getSnapshot().activeProject.id;

    store.commands.gallery.setSearchTerm('positional form', projectId);

    expect(getProjectWidgetValues(store.getSnapshot().activeProject, 'gallery').searchTerm).toBe('positional form');
  });

  it('dispatches nullary commands with no payload', () => {
    const store = createWorkbenchStore();

    store.commands.notifications.add({ kind: 'info', title: 'To clear' });
    store.commands.notifications.clear();

    expect(store.getSnapshot().notifications).toHaveLength(0);
  });

  it('records recordError actions into project diagnostics', () => {
    const store = createWorkbenchStore();
    const projectId = store.getSnapshot().activeProject.id;

    clearProjectDiagnostics(projectId);
    store.commands.notifications.reportError({
      area: 'queue-runtime',
      message: 'Queue failed',
      namespace: 'queue',
    });

    expect(getProjectDiagnostics(projectId)).toMatchObject([
      {
        level: 'error',
        message: 'Queue failed',
        namespace: 'queue',
        source: { area: 'queue-runtime', kind: 'workbench', projectId },
      },
    ]);
    expect(store.getSnapshot().notifications[0]?.title).toBe('Error');
  });

  it('records accepted widget failures into diagnostics once', () => {
    const store = createWorkbenchStore();
    const projectId = store.getSnapshot().activeProject.id;
    const failure = {
      details: 'Widget stack',
      message: 'Widget failed',
      occurredAt: '2026-06-29T00:00:00.000Z',
      widgetId: 'workflow' as const,
    };

    clearProjectDiagnostics(projectId);
    store.commands.notifications.recordWidgetFailure(failure);
    store.commands.notifications.recordWidgetFailure(failure);

    expect(getProjectDiagnostics(projectId)).toMatchObject([
      {
        context: { widgetId: 'workflow' },
        level: 'error',
        message: 'Widget stack',
        namespace: 'system',
        source: { area: 'widget-failure', kind: 'workbench', projectId },
      },
    ]);
  });

  it('stops notifying unsubscribed listeners', () => {
    const store = createWorkbenchStore();
    const listener = vi.fn();
    const unsubscribe = store.subscribe(listener);

    unsubscribe();
    store.commands.projects.create();

    expect(listener).not.toHaveBeenCalled();
  });

  it('keeps active-project selectors stable across unrelated global state updates', () => {
    const store = createWorkbenchStore();
    const activeProjectWatcher = watchSelector(store, (snapshot) => snapshot.activeProject);
    const activeProjectIdWatcher = watchSelector(store, (snapshot) => snapshot.activeProject.id);
    const notificationCountWatcher = watchSelector(store, (snapshot) => snapshot.notifications.length);

    store.commands.notifications.add({ kind: 'info', title: 'Global notice' });

    expect(notificationCountWatcher.current).toBe(1);
    expect(notificationCountWatcher.changeCount).toBe(1);
    expect(activeProjectWatcher.changeCount).toBe(0);
    expect(activeProjectIdWatcher.changeCount).toBe(0);
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

  it('supports custom equality for derived selector objects', () => {
    const store = createWorkbenchStore();
    const statusObjectWatcher = watchSelector(
      store,
      (snapshot) => ({ status: snapshot.backendConnection.status }),
      (left, right) => left.status === right.status
    );

    store.commands.notifications.add({ kind: 'info', title: 'Unrelated notice' });

    expect(statusObjectWatcher.current).toEqual({ status: 'connecting' });
    expect(statusObjectWatcher.changeCount).toBe(0);

    store.commands.queue.setConnectionStatus({ status: 'connected' });

    expect(statusObjectWatcher.current).toEqual({ status: 'connected' });
    expect(statusObjectWatcher.changeCount).toBe(1);
  });

  it('keeps widget placement metadata stable across widget value updates', () => {
    const store = createWorkbenchStore();
    const placementWatcher = watchSelector(
      store,
      (snapshot) => getWidgetPlacementProject(snapshot.activeProject),
      areWidgetPlacementProjectsEqual
    );
    const widgetInstancesWatcher = watchSelector(store, (snapshot) => snapshot.activeProject.widgetInstances);

    store.commands.widgets.patchInstanceValues('generate', { prompt: 'updated' });

    expect(widgetInstancesWatcher.changeCount).toBe(1);
    expect(placementWatcher.changeCount).toBe(0);
  });

  it('treats identical widget placement in a different project as a placement change', () => {
    const store = createWorkbenchStore();
    const placementWatcher = watchSelector(
      store,
      (snapshot) => getWidgetPlacementProject(snapshot.activeProject),
      areWidgetPlacementProjectsEqual
    );
    const firstProjectId = placementWatcher.current.projectId;

    store.commands.projects.create();

    expect(placementWatcher.current.projectId).not.toBe(firstProjectId);
    expect(placementWatcher.changeCount).toBe(1);
  });

  it('does not notify subscribers for equivalent generate and project settings writes', () => {
    const store = createWorkbenchStore();
    const listener = vi.fn();

    store.subscribe(listener);

    store.commands.generation.setBatchCount(1);
    store.commands.account.updateProjectPreferences({ useCpuNoise: true });
    store.commands.layout.setRegionCollapsed('left', false);
    store.commands.layout.setRegionSize('left', store.getSnapshot().activeProject.widgetRegions.left.sizePx);
    store.commands.widgets.patchValues('generate', {});
    store.commands.queue.setConnectionStatus({ status: 'connecting' });

    expect(listener).not.toHaveBeenCalled();
  });

  it('does not notify subscribers for equivalent queue status writes', () => {
    const store = createWorkbenchStore();
    const project = store.getSnapshot().activeProject;
    const listener = vi.fn();

    store.subscribe(listener);

    store.commands.queue.setStatus({ projectId: project.id, queueItemId: 'missing-queue-item', status: 'pending' });

    expect(listener).not.toHaveBeenCalled();
  });

  it('exposes queue behavior through namespaced commands', () => {
    const initialState = createInitialWorkbenchState();
    const initialProject = initialState.projects[0];
    const store = createWorkbenchStore({
      ...initialState,
      projects: [
        {
          ...initialProject,
          queue: {
            items: [
              {
                cancellable: true,
                id: 'queue-item-1',
                snapshot: {} as (typeof initialProject.queue.items)[number]['snapshot'],
                status: 'pending',
              },
              {
                cancellable: false,
                id: 'queue-item-complete',
                snapshot: {} as (typeof initialProject.queue.items)[number]['snapshot'],
                status: 'completed',
              },
            ],
          },
        },
      ],
    });
    const projectId = store.getSnapshot().activeProject.id;

    store.commands.queue.setConnectionStatus({ status: 'connected' });
    store.commands.queue.cancel(projectId, 'queue-item-1');

    expect(store.getState().backendConnection.status).toBe('connected');
    expect(store.getSnapshot().activeProject.queue.items[0]?.status).toBe('cancelled');
    expect(store.getSnapshot().notifications[0]).toMatchObject({
      kind: 'info',
      title: 'Invocation cancellation requested',
    });

    store.commands.queue.clearCompleted();

    expect(store.getSnapshot().activeProject.queue.items.map((item) => item.id)).toEqual(['queue-item-1']);
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
    expect(store.getSnapshot().notifications[0]).toMatchObject({
      kind: 'error',
      title: 'Project close blocked',
    });
  });

  it('applies Canvas and Workflow edits without exposing aggregate reducer actions', () => {
    const store = createWorkbenchStore();
    const project = store.getSnapshot().activeProject;
    const bbox = project.canvas.document.bbox;

    expect(store.commands.canvas.apply(project.id, { bbox: { ...bbox, x: bbox.x + 10 }, type: 'setCanvasBbox' })).toBe(
      true
    );
    expect(store.getSnapshot().activeProject.canvas.document.bbox.x).toBe(bbox.x + 10);

    store.commands.workflows.editGraph({ patch: { name: 'Command-owned workflow' }, type: 'setMetadata' });

    expect(store.getSnapshot().activeProject.projectGraph.name).toBe('Command-owned workflow');
  });

  it('keeps placement selectors stable across generate and settings changes', () => {
    const store = createWorkbenchStore();
    const placementWatcher = watchSelector(
      store,
      (snapshot) => getWidgetPlacementProject(snapshot.activeProject),
      areWidgetPlacementProjectsEqual
    );

    store.commands.generation.setBatchCount(2);
    store.commands.account.updateProjectPreferences({ useCpuNoise: false });

    expect(placementWatcher.changeCount).toBe(0);
  });
});
