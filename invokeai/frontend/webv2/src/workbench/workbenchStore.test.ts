import { describe, expect, it, vi } from 'vitest';

import type { WorkbenchSnapshot, WorkbenchStore } from './workbenchStore';

import { areWidgetPlacementProjectsEqual, getWidgetPlacementProject } from './widgetPlacementMeta';
import { getProjectWidgetValues } from './widgetState';
import { createWorkbenchStore } from './workbenchStore';

const watchSelector = <Selected>(
  store: WorkbenchStore,
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
  it('exposes a stable dispatch and initializes snapshot metadata', () => {
    const store = createWorkbenchStore();
    const snapshot = store.getSnapshot();

    expect(store.dispatch).toBe(store.dispatch);
    expect(snapshot.state.activeProjectId).toBe(snapshot.activeProject.id);
    expect(snapshot.hasHydrated).toBe(false);
    expect(snapshot.state.projects).toHaveLength(1);
  });

  it('notifies subscribers once for reducer changes and not for no-op reducer results', () => {
    const store = createWorkbenchStore();
    const listener = vi.fn();

    store.subscribe(listener);

    const projectId = store.getSnapshot().activeProject.id;

    store.dispatch({ name: 'Renamed', projectId, type: 'renameProject' });

    expect(listener).toHaveBeenCalledTimes(1);
    expect(store.getSnapshot().activeProject.name).toBe('Renamed');

    store.dispatch({ name: '   ', projectId, type: 'renameProject' });

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

    store.dispatch({ status: 'connected', type: 'setBackendConnectionStatus' });

    expect(store.getPersistedRevision()).toBe(initialPersistedRevision);

    store.dispatch({ kind: 'info', title: 'Global notice', type: 'recordNotice' });

    expect(store.getPersistedRevision()).toBe(initialPersistedRevision);

    store.dispatch({
      name: 'Persisted rename',
      projectId: store.getSnapshot().activeProject.id,
      type: 'renameProject',
    });

    expect(store.getPersistedRevision()).toBe(initialPersistedRevision + 1);
  });

  it('stops notifying unsubscribed listeners', () => {
    const store = createWorkbenchStore();
    const listener = vi.fn();
    const unsubscribe = store.subscribe(listener);

    unsubscribe();
    store.dispatch({ type: 'createProject' });

    expect(listener).not.toHaveBeenCalled();
  });

  it('keeps active-project selectors stable across unrelated global state updates', () => {
    const store = createWorkbenchStore();
    const activeProjectWatcher = watchSelector(store, (snapshot) => snapshot.activeProject);
    const activeProjectIdWatcher = watchSelector(store, (snapshot) => snapshot.activeProject.id);
    const notificationCountWatcher = watchSelector(store, (snapshot) => snapshot.state.notifications.length);

    store.dispatch({ kind: 'info', title: 'Global notice', type: 'recordNotice' });

    expect(notificationCountWatcher.current).toBe(1);
    expect(notificationCountWatcher.changeCount).toBe(1);
    expect(activeProjectWatcher.changeCount).toBe(0);
    expect(activeProjectIdWatcher.changeCount).toBe(0);
  });

  it('lets narrow selectors observe only their own state changes', () => {
    const store = createWorkbenchStore();
    const backendStatusWatcher = watchSelector(store, (snapshot) => snapshot.state.backendConnection.status);
    const activeProjectIdWatcher = watchSelector(store, (snapshot) => snapshot.activeProject.id);
    const projectCountWatcher = watchSelector(store, (snapshot) => snapshot.state.projects.length);
    const batchCountWatcher = watchSelector(store, (snapshot) =>
      Number(getProjectWidgetValues(snapshot.activeProject, 'generate').batchCount ?? 1)
    );

    store.dispatch({ status: 'connected', type: 'setBackendConnectionStatus' });

    expect(backendStatusWatcher.current).toBe('connected');
    expect(backendStatusWatcher.changeCount).toBe(1);
    expect(activeProjectIdWatcher.changeCount).toBe(0);
    expect(projectCountWatcher.changeCount).toBe(0);
    expect(batchCountWatcher.changeCount).toBe(0);

    store.dispatch({ batchCount: 3, type: 'setGenerateBatchCount' });

    expect(batchCountWatcher.current).toBe(3);
    expect(batchCountWatcher.changeCount).toBe(1);
    expect(backendStatusWatcher.changeCount).toBe(1);
    expect(activeProjectIdWatcher.changeCount).toBe(0);
    expect(projectCountWatcher.changeCount).toBe(0);

    store.dispatch({ type: 'createProject' });

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
      (snapshot) => ({ status: snapshot.state.backendConnection.status }),
      (left, right) => left.status === right.status
    );

    store.dispatch({ kind: 'info', title: 'Unrelated notice', type: 'recordNotice' });

    expect(statusObjectWatcher.current).toEqual({ status: 'connecting' });
    expect(statusObjectWatcher.changeCount).toBe(0);

    store.dispatch({ status: 'connected', type: 'setBackendConnectionStatus' });

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

    store.dispatch({ instanceId: 'generate', type: 'patchWidgetInstanceValues', values: { prompt: 'updated' } });

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

    store.dispatch({ type: 'createProject' });

    expect(placementWatcher.current.projectId).not.toBe(firstProjectId);
    expect(placementWatcher.changeCount).toBe(1);
  });

  it('does not notify subscribers for equivalent generate and project settings writes', () => {
    const store = createWorkbenchStore();
    const listener = vi.fn();

    store.subscribe(listener);

    store.dispatch({ batchCount: 1, type: 'setGenerateBatchCount' });
    store.dispatch({ settings: { useCpuNoise: true }, type: 'setActiveProjectSettings' });
    store.dispatch({ isCollapsed: false, region: 'left', type: 'setRegionWidgetCollapsed' });
    store.dispatch({
      region: 'left',
      sizePx: store.getSnapshot().activeProject.widgetRegions.left.sizePx,
      type: 'setRegionWidgetSize',
    });
    store.dispatch({ type: 'patchWidgetValues', values: {}, widgetId: 'generate' });
    store.dispatch({ status: 'connecting', type: 'setBackendConnectionStatus' });

    expect(listener).not.toHaveBeenCalled();
  });

  it('does not notify subscribers for equivalent queue status writes', () => {
    const store = createWorkbenchStore();
    const project = store.getSnapshot().activeProject;
    const listener = vi.fn();

    store.subscribe(listener);

    store.dispatch({
      projectId: project.id,
      queueItemId: 'missing-queue-item',
      status: 'pending',
      type: 'setQueueItemStatus',
    });

    expect(listener).not.toHaveBeenCalled();
  });

  it('keeps placement selectors stable across generate, queue, and settings changes', () => {
    const store = createWorkbenchStore();
    const placementWatcher = watchSelector(
      store,
      (snapshot) => getWidgetPlacementProject(snapshot.activeProject),
      areWidgetPlacementProjectsEqual
    );

    store.dispatch({ batchCount: 2, type: 'setGenerateBatchCount' });
    store.dispatch({ backendSupportsCancellation: true, type: 'submitInvocationSnapshot' });
    store.dispatch({ settings: { useCpuNoise: false }, type: 'setActiveProjectSettings' });

    expect(placementWatcher.changeCount).toBe(0);
  });
});
