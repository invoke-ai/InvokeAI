import { createContext, useContext } from 'react';
import { assert } from 'tsafe';

export const RefImageIdContext = createContext<string | null>(null);

export const useRefImageIdContext = (): string => {
  const id = useContext(RefImageIdContext);
  assert(id, 'useRefImageIdContext must be used within a RefImageIdContext.Provider');
  return id;
};
