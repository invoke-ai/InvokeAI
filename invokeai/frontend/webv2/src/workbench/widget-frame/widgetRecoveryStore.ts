import type { RegisteredWidget } from '@workbench/widgetContracts';

interface Recovery {
  generation: number;
  retrying: boolean;
  reloading: boolean;
  reloadAvailable: boolean;
  errors: Readonly<Record<string, Error>>;
}

/** Per-instance coordination; each render boundary still owns its failing subtree. */
export const createWidgetRecovery = () => {
  let snapshot: Recovery = { generation: 0, retrying: false, reloading: false, reloadAvailable: false, errors: {} };
  const listeners = new Set<() => void>();
  const mounted = new Map<string, number>();
  const failedGeneration = new Map<string, number>();
  const publish = (next: Recovery) => {
    snapshot = next;
    for (const listener of listeners) {
      listener();
    }
  };
  const retireInactiveFailures = (errors: Recovery['errors']) =>
    Object.fromEntries(
      Object.entries(errors).filter(([slot]) => mounted.has(slot) || failedGeneration.get(slot) === snapshot.generation)
    );
  let pending = false;
  return {
    getSnapshot: () => snapshot,
    subscribe: (listener: () => void, slot: string) => {
      listeners.add(listener);
      mounted.set(slot, (mounted.get(slot) ?? 0) + 1);
      return () => {
        listeners.delete(listener);
        const count = mounted.get(slot)! - 1;
        if (count) {
          mounted.set(slot, count);
        } else {
          mounted.delete(slot);
        }
      };
    },
    fail: (slot: string, error: Error) => {
      if (snapshot.errors[slot] === error && failedGeneration.get(slot) === snapshot.generation) {
        return false;
      }
      const first = !Object.values(snapshot.errors).includes(error);
      failedGeneration.set(slot, snapshot.generation);
      publish({ ...snapshot, errors: { ...snapshot.errors, [slot]: error } });
      return first;
    },
    recover: (slot: string) => {
      if (!snapshot.errors[slot]) {
        return;
      }
      const { [slot]: _error, ...errors } = snapshot.errors;
      failedGeneration.delete(slot);
      publish({ ...snapshot, errors });
    },
    retry: (widget: RegisteredWidget) => {
      if (pending) {
        return;
      }
      pending = true;
      void widget.implementation
        .retry()
        .catch(() => publish({ ...snapshot, reloadAvailable: true }))
        .finally(() => {
          pending = false;
          publish({ ...snapshot, retrying: false, errors: retireInactiveFailures(snapshot.errors) });
        });
      publish({ ...snapshot, generation: snapshot.generation + 1, retrying: true, reloadAvailable: false });
    },
    reload: async (run: () => Promise<void>) => {
      if (pending) {
        return;
      }
      pending = true;
      publish({ ...snapshot, reloading: true });
      try {
        await run();
      } finally {
        pending = false;
        publish({ ...snapshot, reloading: false });
      }
    },
  };
};
export type WidgetRecoveryStore = ReturnType<typeof createWidgetRecovery>;
