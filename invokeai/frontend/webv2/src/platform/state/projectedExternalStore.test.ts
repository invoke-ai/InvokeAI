import { describe, expect, it, vi } from 'vitest';

import { createProjectedExternalStore } from './projectedExternalStore';
import { shallowEqual } from './selectors';

describe('createProjectedExternalStore', () => {
  it('keeps projected snapshots stable and silent for unrelated source changes', () => {
    let source = {
      modelsRevision: 0,
      notificationsRevision: 0,
      projectId: 'project-1',
      routeRevision: 0,
      sessionRevision: 0,
    };
    const sourceListeners = new Set<() => void>();
    const store = createProjectedExternalStore({
      source: {
        getSnapshot: () => source,
        subscribe: (listener) => {
          sourceListeners.add(listener);
          return () => sourceListeners.delete(listener);
        },
      },
      select: (snapshot) => ({ id: snapshot.projectId }),
      isEqual: shallowEqual,
    });
    const listener = vi.fn();
    store.subscribe(listener);
    const initial = store.getSnapshot();

    source = { ...source, modelsRevision: 1, notificationsRevision: 1, routeRevision: 1, sessionRevision: 1 };
    sourceListeners.forEach((notify) => notify());

    expect(listener).not.toHaveBeenCalled();
    expect(store.getSnapshot()).toBe(initial);

    source = { ...source, projectId: 'project-2' };
    sourceListeners.forEach((notify) => notify());

    expect(listener).toHaveBeenCalledOnce();
    expect(store.getSnapshot()).toEqual({ id: 'project-2' });
    expect(store.getSnapshot()).toBe(store.getSnapshot());
  });

  it('shares one lazy source subscription across every projected-store subscriber', () => {
    let source = { projectId: 'project-1' };
    const sourceListeners = new Set<() => void>();
    const store = createProjectedExternalStore({
      source: {
        getSnapshot: () => source,
        subscribe: (listener) => {
          sourceListeners.add(listener);
          return () => sourceListeners.delete(listener);
        },
      },
      select: (snapshot) => ({ id: snapshot.projectId }),
      isEqual: shallowEqual,
    });
    const firstListener = vi.fn();
    const secondListener = vi.fn();
    const unsubscribeFirst = store.subscribe(firstListener);
    const unsubscribeSecond = store.subscribe(secondListener);

    expect(sourceListeners.size).toBe(1);

    source = { projectId: 'project-2' };
    sourceListeners.forEach((notify) => notify());

    expect(firstListener).toHaveBeenCalledOnce();
    expect(secondListener).toHaveBeenCalledOnce();

    unsubscribeFirst();
    unsubscribeSecond();
    expect(sourceListeners.size).toBe(0);
  });

  it('refreshes on demand after its final source subscription is removed', () => {
    let source = { projectId: 'project-1' };
    const sourceListeners = new Set<() => void>();
    const store = createProjectedExternalStore({
      source: {
        getSnapshot: () => source,
        subscribe: (listener) => {
          sourceListeners.add(listener);
          return () => sourceListeners.delete(listener);
        },
      },
      select: (snapshot) => snapshot.projectId,
    });
    const unsubscribe = store.subscribe(vi.fn());
    unsubscribe();

    source = { projectId: 'project-2' };

    expect(sourceListeners.size).toBe(0);
    expect(store.getSnapshot()).toBe('project-2');
  });
});
