import { createContext, useContext } from 'react';

const ClearStorageContext = createContext<() => void>(() => {});

export const ClearStorageProvider = ClearStorageContext.Provider;

export const useClearStorage = () => {
  const context = useContext(ClearStorageContext);
  return context;
};
