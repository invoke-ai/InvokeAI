import { useCallback, useSyncExternalStore } from 'react';

import {
  createCollectionStoreCore,
  createExternalStoreCore,
  createKeyedTransientStoreCore,
  type CollectionStoreCore,
  type ExternalStoreCore,
  type KeyedTransientStoreCore,
} from './externalStoreCore';
import { shallowEqual, useExternalStoreSelector, type EqualityFn } from './selectors';

export * from './externalStoreCore';

export interface ExternalStore<Snapshot extends object> extends ExternalStoreCore<Snapshot> {
  useSelector: <Selected>(selector: (snapshot: Snapshot) => Selected, isEqual?: EqualityFn<Selected>) => Selected;
  useSnapshot: () => Snapshot;
}

export const createExternalStore = <Snapshot extends object>(initialSnapshot: Snapshot): ExternalStore<Snapshot> => {
  const store = createExternalStoreCore(initialSnapshot);
  return {
    ...store,
    useSelector: <Selected>(
      selector: (snapshot: Snapshot) => Selected,
      isEqual: EqualityFn<Selected> = shallowEqual
    ): Selected => useExternalStoreSelector(store.subscribe, store.getSnapshot, selector, isEqual),
    useSnapshot: () => useSyncExternalStore(store.subscribe, store.getSnapshot, store.getSnapshot),
  };
};

export interface KeyedTransientStore<Key, Value> extends KeyedTransientStoreCore<Key, Value> {
  useValue: (key: Key) => Value | undefined;
}

export const createKeyedTransientStore = <Key, Value>(
  isEqual: EqualityFn<Value | undefined> = Object.is
): KeyedTransientStore<Key, Value> => {
  const store = createKeyedTransientStoreCore<Key, Value>(isEqual);
  return {
    ...store,
    useValue: (key) => {
      const subscribe = useCallback((listener: () => void) => store.subscribeKey(key, listener), [key]);
      const getSnapshot = useCallback(() => store.get(key), [key]);
      return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
    },
  };
};

export interface CollectionStore<Contribution> extends CollectionStoreCore<Contribution> {
  useList: () => Contribution[];
}

export const createCollectionStore = <Contribution>(): CollectionStore<Contribution> => {
  const store = createCollectionStoreCore<Contribution>();
  return { ...store, useList: () => useSyncExternalStore(store.subscribe, store.list, store.list) };
};
