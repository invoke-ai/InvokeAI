import type { GridviewApi } from 'dockview';
import type { Atom } from 'nanostores';
import type { PropsWithChildren } from 'react';
import { createContext, useContext } from 'react';

const AutoLayoutContext = createContext<Atom<GridviewApi | null> | null>(null);

export const AutoLayoutProvider = (props: PropsWithChildren<{ $api: Atom<GridviewApi | null> }>) => {
  return <AutoLayoutContext.Provider value={props.$api}>{props.children}</AutoLayoutContext.Provider>;
};

export const useAutoLayoutContext = () => {
  const api = useContext(AutoLayoutContext);
  if (!api) {
    throw new Error('useAutoLayoutContext must be used within an AutoLayoutProvider');
  }
  return api;
};
