import type { createStore } from 'app/store/store';
import { atom } from 'nanostores';

// Inject socket options and url into window for debugging
declare global {
  interface Window {
    $store?: typeof $store;
  }
}

/**
 * Raised when the redux store is unable to be retrieved.
 */
export class ReduxStoreNotInitialized extends Error {
  /**
   * Create ReduxStoreNotInitialized
   * @param {String} message
   */
  constructor(message = 'Redux store not initialized') {
    super(message);
    this.name = this.constructor.name;
  }
}

export const $store = atom<Readonly<ReturnType<typeof createStore>> | undefined>();

export const getStore = () => {
  const store = $store.get();
  if (!store) {
    throw new ReduxStoreNotInitialized();
  }
  return store;
};
