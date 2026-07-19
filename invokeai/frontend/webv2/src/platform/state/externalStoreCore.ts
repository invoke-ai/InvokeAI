import { shallowEqual, type EqualityFn } from './selectorCore';

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
      return () => listeners.delete(listener);
    },
  };
};

export interface ExternalStoreCore<Snapshot extends object> extends ListenerChannel {
  getSnapshot: () => Snapshot;
  patchSnapshot: (next: Partial<Snapshot>) => void;
  setSnapshot: (next: Snapshot) => void;
  setSnapshotSilently: (next: Snapshot) => void;
}

const hasPatchChanges = <Snapshot extends object>(snapshot: Snapshot, patch: Partial<Snapshot>): boolean =>
  Object.entries(patch).some(
    ([key, value]) => !Object.hasOwn(snapshot, key) || !Object.is(snapshot[key as keyof Snapshot], value)
  );

export const createExternalStoreCore = <Snapshot extends object>(
  initialSnapshot: Snapshot
): ExternalStoreCore<Snapshot> => {
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

  return {
    ...channel,
    getSnapshot,
    patchSnapshot: (next) => {
      if (hasPatchChanges(snapshot, next)) {
        setSnapshot({ ...snapshot, ...next });
      }
    },
    setSnapshot,
    setSnapshotSilently: (next) => {
      snapshot = next;
    },
  };
};

export interface KeyedTransientStoreCore<Key, Value> extends ListenerChannel {
  clear: () => void;
  delete: (key: Key) => void;
  entries: () => Array<[Key, Value]>;
  get: (key: Key) => Value | undefined;
  set: (key: Key, value: Value) => void;
  subscribeKey: (key: Key, listener: () => void) => () => void;
}

export const createKeyedTransientStoreCore = <Key, Value>(
  isEqual: EqualityFn<Value | undefined> = Object.is
): KeyedTransientStoreCore<Key, Value> => {
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
      if (!values.has(key) || !isEqual(values.get(key), value)) {
        values.set(key, value);
        notifyKey(key);
      }
    },
    subscribeKey,
  };
};

export interface CollectionStoreCore<Contribution> extends ListenerChannel {
  findLatest: (predicate: (contribution: Contribution) => boolean) => Contribution | undefined;
  list: () => Contribution[];
  register: (contribution: Contribution, registrationKey: string) => () => void;
}

export const createCollectionStoreCore = <Contribution>(): CollectionStoreCore<Contribution> => {
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
      const token = nextRegistrationId;
      nextRegistrationId += 1;
      contributions.delete(registrationKey);
      contributions.set(registrationKey, { contribution, token });
      refreshListSnapshot();
      channel.notify();
      return () => {
        if (contributions.get(registrationKey)?.token === token && contributions.delete(registrationKey)) {
          refreshListSnapshot();
          channel.notify();
        }
      };
    },
  };
};
