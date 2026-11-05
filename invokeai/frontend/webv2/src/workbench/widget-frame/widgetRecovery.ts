import type { RegisteredWidget } from '@workbench/widgetContracts';

import { useActiveProjectSelector, useWorkbenchInternalStore } from '@workbench/WorkbenchContext';
import { useMemo, useSyncExternalStore } from 'react';

interface Recovery {
  generation: number;
  error?: Error;
}

export const createWidgetRecovery = (widget: RegisteredWidget) => {
  let snapshot: Recovery = { generation: 0 };
  const listeners = new Set<() => void>();
  const publish = (next: Recovery) => {
    snapshot = next;
    for (const listener of listeners) {
      listener();
    }
  };
  let retrying = false;
  return {
    getSnapshot: () => snapshot,
    subscribe: (listener: () => void) => {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
    fail: (error: Error) => {
      if (!snapshot.error) {
        publish({ ...snapshot, error });
      }
    },
    retry: () => {
      if (retrying) {
        return;
      }
      retrying = true;
      void Promise.resolve(widget.implementation.retry())
        .catch(() => undefined)
        .finally(() => {
          retrying = false;
        });
      publish({ generation: snapshot.generation + 1 });
    },
  };
};

export type WidgetRecoveryStore = ReturnType<typeof createWidgetRecovery>;

export const useWidgetRecovery = (instanceId: string, widget: RegisteredWidget) => {
  const owner = useWorkbenchInternalStore();
  const projectId = useActiveProjectSelector((project) => project.id);
  const recovery = useMemo(() => {
    const entries = owner.widgetRecovery;
    const key = JSON.stringify([projectId, instanceId]);
    let entry = entries.get(key);
    if (!entry) {
      entry = createWidgetRecovery(widget);
      entries.set(key, entry);
    }
    return entry;
  }, [owner, projectId, instanceId, widget]);
  const snapshot = useSyncExternalStore(recovery.subscribe, recovery.getSnapshot);
  const isCurrent = () => {
    const project = owner.getSnapshot().activeProject;
    return (
      project.id === projectId &&
      !!project.widgetInstances[instanceId] &&
      owner.widgetRecovery.get(JSON.stringify([projectId, instanceId])) === recovery
    );
  };
  return {
    ...snapshot,
    retry: () => {
      if (isCurrent()) {
        recovery.retry();
      }
    },
    fail: (error: Error) => {
      if (isCurrent() && recovery.getSnapshot().generation === snapshot.generation) {
        recovery.fail(error);
      }
    },
  };
};
