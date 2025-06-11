import type { GridviewApi } from 'dockview';
import type { PropsWithChildren } from 'react';
import { createContext, useContext } from 'react';

const AutoLayoutContext = createContext<GridviewApi | null>(null);

export const AutoLayoutProvider = (props: PropsWithChildren<{ api: GridviewApi | null }>) => {
  return <AutoLayoutContext.Provider value={props.api}>{props.children}</AutoLayoutContext.Provider>;
};

export const useAutoLayoutContext = () => {
  const api = useContext(AutoLayoutContext);
  return api;
};
