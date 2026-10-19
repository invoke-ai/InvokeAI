import { useSyncExternalStoreWithSelector } from 'use-sync-external-store/with-selector';

import { shallowEqual, type EqualityFn } from './selectorCore';

export * from './selectorCore';

/** React adapter for a React-free external-store subscription. */
export const useExternalStoreSelector = <Snapshot, Selected>(
  subscribe: (listener: () => void) => () => void,
  getSnapshot: () => Snapshot,
  selector: (snapshot: Snapshot) => Selected,
  isEqual: EqualityFn<Selected> = shallowEqual
): Selected => useSyncExternalStoreWithSelector(subscribe, getSnapshot, getSnapshot, selector, isEqual);
