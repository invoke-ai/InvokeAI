import { describe, expect, it, vi } from 'vitest';

import type { WorkbenchSnapshot, WorkbenchStore } from './workbenchStore';

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
    const listener = vi.fn();

    store.subscribe(listener);

    store.setHasHydrated(true);

    expect(listener).toHaveBeenCalledTimes(1);
    expect(store.getSnapshot().hasHydrated).toBe(true);
    expect(store.getState()).toBe(initialState);

    store.setHasHydrated(true);

    expect(listener).toHaveBeenCalledTimes(1);
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
      Number(snapshot.activeProject.widgetStates.generate.values.batchCount ?? 1)
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
});
