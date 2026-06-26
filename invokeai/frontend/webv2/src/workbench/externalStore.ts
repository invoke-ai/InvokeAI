import { useCallback, useSyncExternalStore } from 'react';

import { shallowEqual, useExternalStoreSelector, type EqualityFn } from './workbenchSelectors';

/**
 * Minimal factory for module-level stores backed by `useSyncExternalStore`.
 * For backend-owned or session-lived state shared across widget surfaces
 * (model library, install jobs, starter catalog) where the workbench reducer
 * is the wrong home. See `models/modelsStore.ts` for the canonical usage.
 */

/** A bare listener channel, for auxiliary data living beside a snapshot. */
export interface ListenerChannel {
  subscribe: (listener: () => void) => () => void;
  notify: () => void;
}

export const createListenerChannel = (): ListenerChannel => {
  const listeners = new Set<() => void>();

  return {
    notify: () => {
      for (const listener of listeners) {
        listener();
      }
    },
    subscribe: (listener) => {
      listeners.add(listener);

      return () => {
        listeners.delete(listener);
      };
    },
  };
};

export interface ExternalStore<Snapshot extends object> extends ListenerChannel {
  getSnapshot: () => Snapshot;
  /** Replace the snapshot and notify subscribers. */
  setSnapshot: (next: Snapshot) => void;
  /** Merge a partial snapshot and notify subscribers. */
  patchSnapshot: (next: Partial<Snapshot>) => void;
  /** Replace without notifying — for data read imperatively, never rendered. */
  setSnapshotSilently: (next: Snapshot) => void;
  /** Subscribe the calling component to the snapshot. */
  useSnapshot: () => Snapshot;
  /** Subscribe the calling component to a selected snapshot value. */
  useSelector: <Selected>(
    selector: (snapshot: Snapshot) => Selected,
    isEqual?: (left: Selected, right: Selected) => boolean
  ) => Selected;
}

const hasPatchChanges = <Snapshot extends object>(snapshot: Snapshot, patch: Partial<Snapshot>): boolean =>
  Object.entries(patch).some(
    ([key, value]) =>
      !Object.prototype.hasOwnProperty.call(snapshot, key) || !Object.is(snapshot[key as keyof Snapshot], value)
  );

export const createExternalStore = <Snapshot extends object>(initialSnapshot: Snapshot): ExternalStore<Snapshot> => {
  let snapshot = initialSnapshot;
  const channel = createListenerChannel();

  const getSnapshot = (): Snapshot => snapshot;

  const setSnapshot = (next: Snapshot): void => {
    if (shallowEqual(next, snapshot)) {
      return;
    }

    snapshot = next;
    channel.notify();
  };

  const useSnapshot = (): Snapshot => useSyncExternalStore(channel.subscribe, getSnapshot);

  const useSelector = <Selected>(
    selector: (snapshot: Snapshot) => Selected,
    isEqual: EqualityFn<Selected> = shallowEqual
  ): Selected => useExternalStoreSelector(channel.subscribe, getSnapshot, selector, isEqual);

  return {
    ...channel,
    getSnapshot,
    patchSnapshot: (next) => {
      if (!hasPatchChanges(snapshot, next)) {
        return;
      }

      setSnapshot({ ...snapshot, ...next });
    },
    setSnapshot,
    setSnapshotSilently: (next) => {
      snapshot = next;
    },
    useSelector,
    useSnapshot,
  };
};

export interface KeyedTransientStore<Key, Value> extends ListenerChannel {
  clear: () => void;
  delete: (key: Key) => void;
  entries: () => Array<[Key, Value]>;
  get: (key: Key) => Value | undefined;
  set: (key: Key, value: Value) => void;
  subscribeKey: (key: Key, listener: () => void) => () => void;
  useValue: (key: Key) => Value | undefined;
}

export const createKeyedTransientStore = <Key, Value>(
  isEqual: EqualityFn<Value | undefined> = Object.is
): KeyedTransientStore<Key, Value> => {
  const values = new Map<Key, Value>();
  const channel = createListenerChannel();
  const keyedListeners = new Map<Key, Set<() => void>>();

  const notifyKey = (key: Key): void => {
    for (const listener of keyedListeners.get(key) ?? []) {
      listener();
    }

    channel.notify();
  };

  const subscribeKey = (key: Key, listener: () => void): (() => void) => {
    const listeners = keyedListeners.get(key) ?? new Set<() => void>();

    listeners.add(listener);
    keyedListeners.set(key, listeners);

    return () => {
      listeners.delete(listener);

      if (listeners.size === 0) {
        keyedListeners.delete(key);
      }
    };
  };

  const get = (key: Key): Value | undefined => values.get(key);

  return {
    ...channel,
    clear: () => {
      if (values.size === 0) {
        return;
      }

      const keys = Array.from(values.keys());
      values.clear();

      for (const key of keys) {
        for (const listener of keyedListeners.get(key) ?? []) {
          listener();
        }
      }

      channel.notify();
    },
    delete: (key) => {
      if (values.delete(key)) {
        notifyKey(key);
      }
    },
    entries: () => [...values.entries()],
    get,
    set: (key, value) => {
      if (values.has(key) && isEqual(values.get(key), value)) {
        return;
      }

      values.set(key, value);
      notifyKey(key);
    },
    subscribeKey,
    useValue: (key) => {
      const subscribe = useCallback((listener: () => void) => subscribeKey(key, listener), [key]);
      const getKeySnapshot = useCallback(() => get(key), [key]);

      return useSyncExternalStore(subscribe, getKeySnapshot);
    },
  };
};

export interface CollectionStore<Contribution> extends ListenerChannel {
  findLatest: (predicate: (contribution: Contribution) => boolean) => Contribution | undefined;
  list: () => Contribution[];
  register: (contribution: Contribution, registrationKey: string) => () => void;
  useList: () => Contribution[];
}

export const createCollectionStore = <Contribution>(): CollectionStore<Contribution> => {
  let nextRegistrationId = 0;
  const contributions = new Map<string, { contribution: Contribution; token: number }>();
  const channel = createListenerChannel();
  let listSnapshot: Contribution[] = [];
  const refreshListSnapshot = (): void => {
    listSnapshot = [...contributions.values()].map((entry) => entry.contribution);
  };
  const list = (): Contribution[] => listSnapshot;

  return {
    ...channel,
    findLatest: (predicate) => {
      const items = list();

      for (let index = items.length - 1; index >= 0; index -= 1) {
        const contribution = items[index];

        if (contribution !== undefined && predicate(contribution)) {
          return contribution;
        }
      }

      return undefined;
    },
    list,
    register: (contribution, registrationKey) => {
      const id = registrationKey;
      const token = nextRegistrationId;

      nextRegistrationId += 1;
      contributions.delete(id);
      contributions.set(id, { contribution, token });
      refreshListSnapshot();
      channel.notify();

      return () => {
        if (contributions.get(id)?.token === token && contributions.delete(id)) {
          refreshListSnapshot();
          channel.notify();
        }
      };
    },
    useList: () => useSyncExternalStore(channel.subscribe, list),
  };
};
