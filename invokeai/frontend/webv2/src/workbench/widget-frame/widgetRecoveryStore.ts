import type { RegisteredWidget } from '@workbench/widgetContracts';

interface Recovery {
  generation: number;
  retrying: boolean;
  errors: Readonly<Record<string, Error>>;
}

/** Per-instance coordination; each render boundary still owns its failing subtree. */
export const createWidgetRecovery = () => {
  let snapshot: Recovery = { generation: 0, retrying: false, errors: {} };
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
    fail: (slot: string, error: Error) => {
      if (snapshot.errors[slot] === error) {
        return false;
      }
      const first = !Object.values(snapshot.errors).includes(error);
      publish({ ...snapshot, errors: { ...snapshot.errors, [slot]: error } });
      return first;
    },
    recover: (slot: string) => {
      if (!snapshot.errors[slot]) {
        return;
      }
      const { [slot]: _error, ...errors } = snapshot.errors;
      publish({ ...snapshot, errors });
    },
    retry: (widget: RegisteredWidget, reload: () => Promise<void>) => {
      if (retrying) {
        return;
      }
      retrying = true;
      void widget.implementation
        .retry()
        .catch(reload)
        .finally(() => {
          retrying = false;
          publish({ ...snapshot, retrying: false });
        });
      publish({ ...snapshot, generation: snapshot.generation + 1, retrying: true });
    },
  };
};
export type WidgetRecoveryStore = ReturnType<typeof createWidgetRecovery>;
