import type { EqualityFn } from './selectorCore';

/** Domain-neutral read-only external store. */
export interface ReadableExternalStore<Snapshot> {
  getSnapshot(): Snapshot;
  subscribe(listener: () => void): () => void;
}

interface ProjectedExternalStoreOptions<Source, Snapshot> {
  source: ReadableExternalStore<Source>;
  select: (source: Source) => Snapshot;
  isEqual?: EqualityFn<Snapshot>;
}

/**
 * Projects a broad external store into a referentially stable read-only store.
 * The source subscription is shared lazily and unrelated source changes remain
 * silent when the projected snapshot is equal.
 */
export const createProjectedExternalStore = <Source, Snapshot>({
  isEqual = Object.is,
  select,
  source,
}: ProjectedExternalStoreOptions<Source, Snapshot>): ReadableExternalStore<Snapshot> => {
  let sourceSnapshot = source.getSnapshot();
  let snapshot = select(sourceSnapshot);
  const listeners = new Set<() => void>();
  let unsubscribeSource: (() => void) | null = null;

  const refresh = (): boolean => {
    const nextSourceSnapshot = source.getSnapshot();
    if (Object.is(nextSourceSnapshot, sourceSnapshot)) {
      return false;
    }
    sourceSnapshot = nextSourceSnapshot;

    const nextSnapshot = select(nextSourceSnapshot);
    if (isEqual(snapshot, nextSnapshot)) {
      return false;
    }
    snapshot = nextSnapshot;
    return true;
  };

  const notifyIfChanged = (): void => {
    if (!refresh()) {
      return;
    }

    for (const listener of listeners) {
      listener();
    }
  };

  return {
    getSnapshot: () => {
      refresh();
      return snapshot;
    },
    subscribe: (listener) => {
      listeners.add(listener);
      unsubscribeSource ??= source.subscribe(notifyIfChanged);

      return () => {
        listeners.delete(listener);
        if (listeners.size === 0) {
          unsubscribeSource?.();
          unsubscribeSource = null;
        }
      };
    },
  };
};
