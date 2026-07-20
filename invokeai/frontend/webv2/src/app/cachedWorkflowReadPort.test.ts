import { shallowEqual } from '@platform/state/selectors';
import { describe, expect, it, vi } from 'vitest';

import { createCachedWorkflowReadPort } from './cachedWorkflowReadPort';

describe('createCachedWorkflowReadPort', () => {
  it('keeps mapped snapshots stable and silent for unrelated source changes', () => {
    let source = {
      modelsRevision: 0,
      notificationsRevision: 0,
      projectId: 'project-1',
      routeRevision: 0,
      sessionRevision: 0,
    };
    const sourceListeners = new Set<() => void>();
    const port = createCachedWorkflowReadPort(
      (listener) => {
        sourceListeners.add(listener);
        return () => sourceListeners.delete(listener);
      },
      () => source,
      (snapshot) => ({ id: snapshot.projectId }),
      shallowEqual
    );
    const listener = vi.fn();
    port.subscribe(listener);
    const initial = port.getSnapshot();

    source = { ...source, modelsRevision: 1, notificationsRevision: 1, routeRevision: 1, sessionRevision: 1 };
    sourceListeners.forEach((notify) => notify());

    expect(listener).not.toHaveBeenCalled();
    expect(port.getSnapshot()).toBe(initial);

    source = { ...source, projectId: 'project-2' };
    sourceListeners.forEach((notify) => notify());

    expect(listener).toHaveBeenCalledOnce();
    expect(port.getSnapshot()).toEqual({ id: 'project-2' });
    expect(port.getSnapshot()).toBe(port.getSnapshot());
  });

  it('notifies every subscriber when one source change refreshes the shared cache', () => {
    let source = { projectId: 'project-1' };
    const sourceListeners = new Set<() => void>();
    const port = createCachedWorkflowReadPort(
      (listener) => {
        sourceListeners.add(listener);
        return () => sourceListeners.delete(listener);
      },
      () => source,
      (snapshot) => ({ id: snapshot.projectId }),
      shallowEqual
    );
    const firstListener = vi.fn();
    const secondListener = vi.fn();
    const unsubscribeFirst = port.subscribe(firstListener);
    const unsubscribeSecond = port.subscribe(secondListener);

    expect(sourceListeners.size).toBe(1);

    source = { projectId: 'project-2' };
    sourceListeners.forEach((notify) => notify());

    expect(firstListener).toHaveBeenCalledOnce();
    expect(secondListener).toHaveBeenCalledOnce();

    unsubscribeFirst();
    unsubscribeSecond();
    expect(sourceListeners.size).toBe(0);
  });
});
