import {
  StorageGetError,
  StorageSetError,
} from 'app/store/enhancers/reduxRemember/errors';
import type { UseStore } from 'idb-keyval';
import {
  clear,
  createStore as createIDBKeyValStore,
  get,
  set,
} from 'idb-keyval';
import { action, atom } from 'nanostores';
import type { Driver } from 'redux-remember';

// Create a custom idb-keyval store (just needed to customize the name)
export const $idbKeyValStore = atom<UseStore>(
  createIDBKeyValStore('invoke', 'invoke-store')
);

export const clearIdbKeyValStore = action($idbKeyValStore, 'clear', (store) => {
  clear(store.get());
});

// Create redux-remember driver, wrapping idb-keyval
export const idbKeyValDriver: Driver = {
  getItem: (key) => {
    try {
      return get(key, $idbKeyValStore.get());
    } catch (err) {
      throw new StorageGetError(key, err);
    }
  },
  setItem: (key, value) => {
    try {
      return set(key, value, $idbKeyValStore.get());
    } catch (err) {
      throw new StorageSetError(key, value, err);
    }
  },
};
