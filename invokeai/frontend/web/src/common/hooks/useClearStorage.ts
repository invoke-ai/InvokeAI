import { idbKeyValStore } from 'app/store/store';
import { clear } from 'idb-keyval';
import { useCallback } from 'react';

export const useClearStorage = () => {
  const clearStorage = useCallback(() => {
    clear(idbKeyValStore);
    localStorage.clear();
  }, []);

  return clearStorage;
};
