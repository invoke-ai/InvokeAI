import { $resetClientState } from 'app/store/enhancers/reduxRemember/driver';
import { useCallback } from 'react';

export const useClearStorage = () => {
  const clearStorage = useCallback(() => {
    $resetClientState.get()();
    localStorage.clear();
  }, []);

  return clearStorage;
};
