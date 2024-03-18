import { clearIdbKeyValStore } from 'app/store/enhancers/reduxRemember/driver';
import { useCallback } from 'react';

export const useClearStorage = () => {
  const clearStorage = useCallback(() => {
    clearIdbKeyValStore();
    localStorage.clear();
  }, []);

  return clearStorage;
};
